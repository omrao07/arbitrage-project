# backend/engine/exec_policies/portfolio_insurance_obpi.py
from __future__ import annotations

import os
import json
import time
import math
import signal
from typing import Dict, Any, Optional

import redis

from backend.bus.streams import hset, publish_stream  # your helpers

# ----------------- Config / Env -----------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# NAV inputs
KEY_NLV          = os.getenv("PORTFOLIO_NLV_KEY", "portfolio:nlv")  # HGET <KEY_NLV> value -> float
KEY_RETURNS      = os.getenv("PORTFOLIO_RET_KEY", "portfolio:return")  # optional running return

# OBPI state & policy keys
KEY_OBPI_STATE   = os.getenv("OBPI_STATE_KEY", "policy:obpi:state")  # floor, peak, init_nlv, last_w
KEY_WEIGHTS      = os.getenv("ALLOCATOR_WEIGHTS_KEY", "strategy:weight")
KEY_META         = os.getenv("STRAT_META_KEY", "strategy:meta")
KEY_ALERTS       = os.getenv("ALERTS_STREAM", "alerts")
KEY_KILL         = os.getenv("KILL_SWITCH_KEY", "policy:kill_switch")

# OBPI parameters
OBPI_M           = float(os.getenv("OBPI_MULTIPLIER", "3.0"))        # typical 2–5
FLOOR_FRAC       = os.getenv("OBPI_FLOOR_FRAC", "0.9")               # 90% of init/trailing
FLOOR_ABS        = os.getenv("OBPI_FLOOR_ABS", "").strip()           # absolute currency floor overrides FRAC if set
USE_TRAILING     = os.getenv("OBPI_USE_TRAIL", "true").lower() in ("1","true","yes","on")  # trail peak NAV
RISK_FREE_APY    = float(os.getenv("OBPI_RISK_FREE_APY", "0.0"))     # annualized, to accrete floor
REBAL_BAND       = float(os.getenv("OBPI_REBAL_BAND", "0.02"))       # only publish if change > 2%
EPS              = 1e-6

# Risky set selection
RISKY_TAG        = os.getenv("OBPI_RISKY_TAG", "").strip().lower()   # if set, only strategies with this tag are "risky"
INCLUDE          = {s for s in os.getenv("OBPI_INCLUDE", "").split(",") if s.strip()}
EXCLUDE          = {s for s in os.getenv("OBPI_EXCLUDE", "").split(",") if s.strip()}

# Safe bucket target
SAFE_STRATEGY    = os.getenv("OBPI_SAFE_STRATEGY", "cash")           # must exist in strategy:meta or UI will just show it
MAX_WEIGHT       = float(os.getenv("OBPI_MAX_WEIGHT", "0.45"))       # per-strategy cap when redistributing
ALLOW_NEG        = os.getenv("OBPI_ALLOW_NEG", "false").lower() in ("1","true","yes","on")

INTERVAL_SEC     = float(os.getenv("OBPI_INTERVAL_SEC", "2.0"))

# ------------------------------------------------

def _hget_float(key: str, field: str = "value", default: float = 0.0) -> float:
    try:
        v = r.hget(key, field)
        if v is None:
            return default
        return float(v)#type: ignore
    except Exception:
        return default

def _hgetall_json(key: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        raw = r.hgetall(key) or {}
        for k, v in raw.items():# type: ignore
            if isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                except Exception:
                    out[k] = v
            else:
                out[k] = v
    except Exception:
        pass
    return out

def _kill_on() -> bool:
    try:
        v = r.hget(KEY_KILL, "enabled")
        return str(v).lower() in ("1","true","yes","on")
    except Exception:
        return False

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _now_ms() -> int:
    return int(time.time() * 1000)

class OBPI:
    """
    OBPI controller:
      - Maintains floor (absolute or fraction of init/trailing NAV with optional risk-free drift)
      - Computes cushion = (NAV - floor)/NAV
      - Target risky weight = clamp(M * cushion, 0..1)
      - Redistributes current strategy weights:
         * "risky" set receive w_risky in proportion to their current weights (or equal)
         * SAFE_STRATEGY gets 1 - w_risky
      - Writes HSET strategy:weight <name> {"w":..., "ts":..., "src":"obpi"}
      - Emits event to allocator.weights for audit/visualization
    """

    def __init__(self) -> None:
        self.m = max(0.0, OBPI_M)
        self.rebal_band = max(0.0, REBAL_BAND)
        self.rf_apy = max(0.0, RISK_FREE_APY)

        # Load state
        st = _hgetall_json(KEY_OBPI_STATE)
        self.init_nlv = float(st.get("init_nlv") or 0.0)
        self.peak_nlv = float(st.get("peak_nlv") or 0.0)
        self.floor    = float(st.get("floor") or 0.0)
        self.last_wr  = float(st.get("last_wr") or -1.0)  # last risky weight

    # ---------- State helpers ----------
    def _ensure_init(self, nav: float) -> None:
        if self.init_nlv <= 0.0 and nav > 0.0:
            self.init_nlv = nav
            self.peak_nlv = nav
            # initialize floor
            if FLOOR_ABS:
                try:
                    self.floor = float(FLOOR_ABS)
                except Exception:
                    self.floor = float(FLOOR_FRAC) * nav
            else:
                frac = float(FLOOR_FRAC or 0.0)
                self.floor = frac * (self.peak_nlv if USE_TRAILING else self.init_nlv)
            self._persist_state()

    def _rf_growth(self, dt_sec: float) -> None:
        """Accrete floor by risk-free APY over time (continuous comp approx)."""
        if self.rf_apy <= 0.0 or self.floor <= 0.0 or dt_sec <= 0.0:
            return
        # r is APY; dt in years
        years = dt_sec / (365.0 * 24.0 * 3600.0)
        self.floor *= math.exp(self.rf_apy * years)

    def _update_peak_and_floor(self, nav: float, dt_sec: float) -> None:
        if nav > self.peak_nlv:
            self.peak_nlv = nav
        self._rf_growth(dt_sec)
        if not FLOOR_ABS:
            frac = float(FLOOR_FRAC or 0.0)
            target_base = self.peak_nlv if USE_TRAILING else self.init_nlv
            target_floor = frac * target_base
            # never let floor fall (monotone non-decreasing)
            self.floor = max(self.floor, target_floor)

    def _persist_state(self) -> None:
        ts = _now_ms()
        hset(KEY_OBPI_STATE, "init_nlv", self.init_nlv)
        hset(KEY_OBPI_STATE, "peak_nlv", self.peak_nlv)
        hset(KEY_OBPI_STATE, "floor", self.floor)
        hset(KEY_OBPI_STATE, "last_wr", self.last_wr)
        hset(KEY_OBPI_STATE, "ts", ts)

    # ---------- Core math ----------
    def _target_risky_weight(self, nav: float) -> float:
        if nav <= 0.0 or self.floor <= 0.0 or nav <= self.floor:
            return 0.0
        cushion = max(0.0, (nav - self.floor) / nav)  # in [0,1)
        w_risky = _clamp(self.m * cushion, 0.0, 1.0)
        return w_risky

    # ---------- Weight redistribution ----------
    def _load_current_weights(self) -> Dict[str, float]:
        raw = _hgetall_json(KEY_WEIGHTS)
        out: Dict[str, float] = {}
        for name, v in raw.items():
            try:
                # support {"w": x} or plain number
                if isinstance(v, dict):
                    out[name] = float(v.get("w", v.get("value", 0.0)) or 0.0)
                else:
                    out[name] = float(v)
            except Exception:
                continue
        return out

    def _load_meta_tags(self) -> Dict[str, set]:
        meta = _hgetall_json(KEY_META)
        tags_by_name: Dict[str, set] = {}
        for name, m in meta.items():
            tags = m.get("tags") or []
            tags_by_name[name] = set(t.lower() for t in tags)
        return tags_by_name

    def _split_risky_safe(self, weights: Dict[str, float], tags: Dict[str, set]) -> tuple[list[str], list[str]]:
        names = [n for n in weights.keys() if (not INCLUDE or n in INCLUDE) and n not in EXCLUDE]
        risky: list[str] = []
        safe: list[str] = []
        for n in names:
            if n == SAFE_STRATEGY:
                safe.append(n)
            elif RISKY_TAG:
                risky.append(n) if (RISKY_TAG in tags.get(n, set())) else safe.append(n)
            else:
                risky.append(n)  # default: all strategies are risky unless SAFE_STRATEGY
        if SAFE_STRATEGY not in names:
            # we still allocate to SAFE_STRATEGY even if it had no prior weight/meta (UI may show it as synthetic)
            pass
        return risky, safe

    def _redistribute(self, w_risky: float, curr: Dict[str, float]) -> Dict[str, float]:
        tags = self._load_meta_tags()
        risky, safe_names = self._split_risky_safe(curr, tags)

        # base proportions within risky set
        risky_base_sum = sum(max(0.0, curr.get(n, 0.0)) for n in risky) or 1.0
        risky_props = {n: max(0.0, curr.get(n, 0.0)) / risky_base_sum for n in risky}

        out: Dict[str, float] = {}
        # allocate risky bucket
        for n in risky:
            out[n] = _clamp(w_risky * risky_props.get(n, 0.0), -MAX_WEIGHT if ALLOW_NEG else 0.0, MAX_WEIGHT)
        # allocate safe bucket to SAFE_STRATEGY only (simplest)
        out[SAFE_STRATEGY] = _clamp(1.0 - w_risky, 0.0, 1.0)

        # ensure excluded names get zeroed explicitly
        for n in curr.keys():
            if n not in out and n != SAFE_STRATEGY:
                out[n] = 0.0

        # small cleanup
        for n, v in list(out.items()):
            if abs(v) < EPS:
                out[n] = 0.0
        return out

    # ---------- IO ----------
    def publish(self, weights: Dict[str, float], w_risky: float, nav: float) -> None:
        ts = _now_ms()
        # write individual strategy weights
        for name, w in weights.items():
            hset(KEY_WEIGHTS, name, {"w": float(w), "ts": ts, "src": "obpi"})
        # stream for audit & UI
        publish_stream("allocator.weights", {
            "ts_ms": ts,
            "src": "obpi",
            "nav": nav,
            "floor": self.floor,
            "w_risky": w_risky,
            "weights": json.dumps(weights),
        })

    # ---------- Runner ----------
    def run_forever(self) -> None:
        self._running = True
        last_ts = time.time()

        def _stop(*_):
            self._running = False

        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)

        while self._running:
            try:
                if _kill_on():
                    # zero risky, 100% safe
                    zero = {SAFE_STRATEGY: 1.0}
                    self.publish(zero, 0.0, _hget_float(KEY_NLV))
                    time.sleep(INTERVAL_SEC)
                    continue

                nav = _hget_float(KEY_NLV, "value", 0.0)
                if nav <= 0.0:
                    time.sleep(INTERVAL_SEC)
                    continue

                self._ensure_init(nav)
                now = time.time()
                dt = now - last_ts
                last_ts = now

                # trail peak & accrete floor
                self._update_peak_and_floor(nav, dt)

                # compute risky weight
                w_risky = self._target_risky_weight(nav)

                # rebalance band: avoid noise
                if self.last_wr < 0.0 or abs(w_risky - self.last_wr) > self.rebal_band:
                    curr = self._load_current_weights()
                    neww = self._redistribute(w_risky, curr)
                    self.publish(neww, w_risky, nav)
                    self.last_wr = w_risky
                    self._persist_state()

            except Exception as e:
                publish_stream(KEY_ALERTS, {
                    "ts_ms": _now_ms(),
                    "lvl": "error",
                    "src": "obpi",
                    "msg": "obpi_step_failed",
                    "err": str(e)
                })
            time.sleep(INTERVAL_SEC)


# ---------- CLI ----------
if __name__ == "__main__":
    """
    Example:
      export OBPI_MULTIPLIER=3.0 OBPI_FLOOR_FRAC=0.90 OBPI_USE_TRAIL=true OBPI_SAFE_STRATEGY=t_bill
      python -m backend.engine.exec_policies.portfolio_insurance_obpi
    Redis inputs:
      HSET portfolio:nlv value 1000000
      HSET strategy:weight alpha_momo '{"w":0.5}'
      HSET strategy:weight alpha_value '{"w":0.5}'
      HSET strategy:meta   alpha_momo  '{"tags":["momentum","equity"]}'
    """
    OBPI().run_forever()