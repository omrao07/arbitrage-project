# backend/engine/adaptive_latency.py
from __future__ import annotations

import json, math, os, time, threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import redis

"""
Adaptive Latency Controller
---------------------------
Purpose:
  • Track live latencies (RTT, queue delay) & error rates per venue/stream.
  • Produce adaptive (p50/p95) timeouts, pacing intervals, and burst limits.
  • Expose circuit-breaker state with half-open probing.

Integrations:
  • Call `record_sample(venue, rtt_ms, ok=True, q_delay_ms=0)` after each RPC/order/feed tick.
  • Use `timeout_ms(venue)` for per-request timeouts.
  • Use `pacing_interval_ms(venue)` to throttle loops / retries.
  • Use `burst_limit(venue)` to cap concurrent in-flight requests.
  • Check `is_open(venue)` to see if breaker is tripped.

Redis I/O (optional but recommended):
  HSET lat:cfg "<venue>" '{"target_p95_ms":300,"min_timeout_ms":120,"max_timeout_ms":1500,
                           "min_pace_ms":5,"max_pace_ms":250,"alpha":0.12,"beta":0.02,
                           "err_half_life_s":60,"lat_half_life_s":30,"open_threshold":0.12}'
  HSET lat:state "<venue>" '{"p50":..., "p95":..., "q50":..., "err":..., "open":0, "ts":...}'
  HSET lat:advice "<venue>" '{"timeout_ms":..., "pace_ms":..., "burst":..., "state":"CLOSED"}'
  SET  risk:halt 0|1  (shared global kill)

Defaults are sensible when no Redis config is present.

Notes:
  • EWMA + P² quantile estimator for p50/p95 (memory-light, robust).
  • Error decays with exponential half-life; breaker trips when err > open_threshold AND p95 >> budget.
  • Half-open probes every `probe_interval_s` while OPEN.
"""

# ============================ CONFIG ============================
REDIS_HOST = os.getenv("AL_REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("AL_REDIS_PORT", "6379"))

CFG_HK     = os.getenv("AL_CFG_HK", "lat:cfg")
STATE_HK   = os.getenv("AL_STATE_HK", "lat:state")
ADVICE_HK  = os.getenv("AL_ADVICE_HK", "lat:advice")
HALT_KEY   = os.getenv("AL_HALT_KEY", "risk:halt")

# Global fallbacks
DEFAULTS = dict(
    target_p95_ms = 300.0,
    min_timeout_ms= 120.0,
    max_timeout_ms= 1500.0,
    min_pace_ms   = 5.0,
    max_pace_ms   = 250.0,
    alpha         = 0.12,   # EWMA for q_delay
    beta          = 0.02,   # EWMA for headroom
    err_half_life_s = 60.0,
    lat_half_life_s = 30.0,
    open_threshold  = 0.12, # error rate threshold
    probe_interval_s= 3.0,
    base_burst      = 4,
    max_burst       = 16
)

# ============================ Redis ============================
_r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

def _hget_json(hk: str, field: str) -> Optional[dict]:
    raw = _r.hget(hk, field)
    if not raw: return None
    try:
        j = json.loads(raw) # type: ignore
        return j if isinstance(j, dict) else None
    except Exception:
        return None

def _hset_json(hk: str, field: str, obj: dict) -> None:
    try:
        _r.hset(hk, field, json.dumps(obj))
    except Exception:
        pass

def _now_s() -> float:
    return time.time()

# ============================ P² Quantile Estimator ============================
class P2Quantile:
    """
    Jain & Chlamtac P² online quantile estimator.
    Track p50 & p95 with tiny memory footprint.
    """
    def __init__(self, p: float):
        self.p = p
        self.n = 0
        self.q = [0.0]*5  # marker heights
        self.np = [0.0]*5 # desired positions
        self.ni = [0]*5   # actual positions
        self.dn = [0.0]*5

    def add(self, x: float) -> None:
        # Bootstrap first 5 samples with sorted values
        if self.n < 5:
            self.q[self.n] = x
            self.n += 1
            if self.n == 5:
                self.q.sort()
                self.ni = [0,1,2,3,4]
                self.np = [0, 2*self.p, 4*self.p, 2+2*self.p, 4]
                self.dn = [0, self.p/2, self.p, (1+self.p)/2, 1]
            return

        # Find cell k
        k = 0
        if x < self.q[0]:
            self.q[0] = x; k = 0
        elif x >= self.q[4]:
            self.q[4] = x; k = 3
        else:
            while k < 4 and x >= self.q[k+1]: k += 1

        # Increment positions of markers 1..4
        for i in range(1,5): self.ni[i] += (1 if i <= k+1 else 0)
        for i in range(5):   self.np[i] += self.dn[i]

        # Adjust heights
        for i in range(1,4):
            d = self.np[i] - self.ni[i]
            s = 1 if d >= 1 and self.ni[i+1] - self.ni[i] > 1 else (-1 if d <= -1 and self.ni[i-1] - self.ni[i] < -1 else 0)
            if s != 0:
                # parabolic prediction
                qip = self.q[i] + s * ((self.ni[i] - self.ni[i-1] + s)*(self.q[i+1]-self.q[i])/(self.ni[i+1]-self.ni[i]) + (self.ni[i+1]-self.ni[i]-s)*(self.q[i]-self.q[i-1])/(self.ni[i]-self.ni[i-1]))/ (self.ni[i+1] - self.ni[i-1])
                # if parabolic goes out of bounds, use linear
                if qip <= self.q[i-1] or qip >= self.q[i+1]:
                    qip = self.q[i] + s*(self.q[i+s] - self.q[i])/(self.ni[i+s]-self.ni[i])
                self.q[i] = qip
                self.ni[i] += s

    def value(self) -> Optional[float]:
        if self.n < 5:
            return None if self.n == 0 else sorted(self.q[:self.n])[int(max(0,min(self.n-1, math.floor(self.p*(self.n-1)))))]
        return float(self.q[2]) if abs(self.p-0.5) < 1e-9 else (float(self.q[3]) if self.p > 0.5 else float(self.q[1]))

# ============================ EWMA helper ============================
@dataclass
class EWMA:
    val: float
    lam: float  # decay factor per sample
    def update(self, x: float) -> float:
        self.val = (1 - self.lam)*self.val + self.lam*x
        return self.val

# ============================ Venue State ============================
@dataclass
class VenueState:
    p50: P2Quantile
    p95: P2Quantile
    q50: EWMA          # queueing delay EWMA
    err: EWMA          # error rate EWMA (0..1)
    last_update_s: float
    open: bool         # breaker open?
    next_probe_s: float
    advice_timeout_ms: float
    advice_pace_ms: float
    advice_burst: int

# ============================ Controller ============================
class AdaptiveLatency:
    def __init__(self, redis_client: Optional[redis.Redis]=None):
        self.r = redis_client or _r
        self._lock = threading.Lock()
        self._venues: Dict[str, VenueState] = {}

    # ---------- public API ----------
    def record_sample(self, venue: str, rtt_ms: float, ok: bool=True, q_delay_ms: float=0.0) -> None:
        """
        Record one RPC/packet round-trip for 'venue'.
        ok=False counts toward error rate.
        q_delay_ms = time spent queued client-side before send.
        """
        if (self.r.get(HALT_KEY) or "0") == "1": return
        v = venue.upper()
        with self._lock:
            st = self._state(v)
            st.p50.add(max(0.0, rtt_ms))
            st.p95.add(max(0.0, rtt_ms))
            st.q50.update(max(0.0, q_delay_ms))
            st.err.update(0.0 if ok else 1.0)
            st.last_update_s = _now_s()
            self._recalc(v, st)

    def timeout_ms(self, venue: str) -> int:
        st = self._state(venue.upper())
        return int(round(st.advice_timeout_ms))

    def pacing_interval_ms(self, venue: str) -> int:
        st = self._state(venue.upper())
        return int(round(st.advice_pace_ms))

    def burst_limit(self, venue: str) -> int:
        st = self._state(venue.upper())
        return max(1, int(st.advice_burst))

    def is_open(self, venue: str) -> bool:
        st = self._state(venue.upper())
        return bool(st.open)

    # ---------- internals ----------
    def _cfg(self, venue: str) -> dict:
        cfg = _hget_json(CFG_HK, venue) or {}
        out = DEFAULTS.copy()
        out.update({k:v for k,v in cfg.items() if k in out})
        return out

    def _state(self, venue: str) -> VenueState:
        st = self._venues.get(venue)
        if st: return st
        cfg = self._cfg(venue)
        # Convert half-lives to per-sample lambda with rough cadence assumption; default sample every second.
        def lam(half_s: float) -> float:
            return max(0.01, min(0.6, math.log(2)/max(1.0, half_s)))
        st = VenueState(
            p50=P2Quantile(0.5),
            p95=P2Quantile(0.95),
            q50=EWMA(val=0.0, lam=cfg["alpha"]),
            err=EWMA(val=0.0, lam=lam(cfg["err_half_life_s"])),
            last_update_s=_now_s(),
            open=False,
            next_probe_s=0.0,
            advice_timeout_ms=cfg["min_timeout_ms"],
            advice_pace_ms=cfg["min_pace_ms"],
            advice_burst=cfg["base_burst"],
        )
        self._venues[venue] = st
        # Try hydrate prior state
        prev = _hget_json(STATE_HK, venue) or {}
        if prev:
            st.advice_timeout_ms = float(prev.get("timeout_ms", st.advice_timeout_ms))
            st.advice_pace_ms    = float(prev.get("pace_ms", st.advice_pace_ms))
            st.advice_burst      = int(prev.get("burst", st.advice_burst))
            st.open              = bool(int(prev.get("open", 0)))
        return st

    def _recalc(self, venue: str, st: VenueState) -> None:
        cfg = self._cfg(venue)

        p50 = st.p50.value() or cfg["min_timeout_ms"]/2.0
        p95 = st.p95.value() or cfg["min_timeout_ms"]
        q50 = st.q50.val
        err = max(0.0, min(1.0, st.err.val))

        # Dynamic timeout: target 1.2×p95 bounded, add small queue EWMA
        t_ms = 1.2 * p95 + 0.5 * q50
        t_ms = min(cfg["max_timeout_ms"], max(cfg["min_timeout_ms"], t_ms))

        # Pacing: keep send interval proportional to p50 and error;
        # more errors -> slower; clamp to min/max.
        pace = p50 * (1.0 + 2.0*err) * 0.5
        pace = min(cfg["max_pace_ms"], max(cfg["min_pace_ms"], pace))

        # Burst: scale with headroom ratio (target_p95 / observed_p95) and error
        target = cfg["target_p95_ms"]
        headroom = max(0.25, min(2.0, target / max(1.0, p95)))
        burst = int(round(min(cfg["max_burst"], max(1.0, cfg["base_burst"] * headroom * (1.0 - 0.5*err)))))

        # Circuit breaker
        # Open if err high and p95 above target by 50%
        should_open = (err >= cfg["open_threshold"]) and (p95 >= 1.5 * target)
        now = _now_s()
        if st.open:
            # half-open probes after interval
            if now >= st.next_probe_s:
                st.open = False  # try half-open; we'll immediately re-open if next sample fails
            else:
                # While open, harden pacing/timeout
                t_ms = max(t_ms, 1.5 * cfg["target_p95_ms"])
                pace = max(pace, 0.8 * cfg["max_pace_ms"])
                burst = 1
        elif should_open:
            st.open = True
            st.next_probe_s = now + cfg["probe_interval_s"]
            burst = 1
            pace = max(pace, 0.8 * cfg["max_pace_ms"])

        # Save advice
        st.advice_timeout_ms = t_ms
        st.advice_pace_ms = pace
        st.advice_burst = burst

        # Persist state & advice
        _hset_json(STATE_HK, venue, {
            "p50": p50, "p95": p95, "q50": q50, "err": err,
            "open": 1 if st.open else 0, "ts": now
        })
        _hset_json(ADVICE_HK, venue, {
            "timeout_ms": int(round(t_ms)),
            "pace_ms": int(round(pace)),
            "burst": int(burst),
            "state": "OPEN" if st.open else "CLOSED"
        })

# ============================ Convenience Singleton ============================
_controller: Optional[AdaptiveLatency] = None

def controller() -> AdaptiveLatency:
    global _controller
    if _controller is None:
        _controller = AdaptiveLatency()
    return _controller

# ============================ Example usage (optional) ============================
if __name__ == "__main__":
    al = controller()
    VEN = "BINANCE_WS"

    # Simulate samples
    for i in range(200):
        rtt = 40 + 20*math.sin(i/20.0) + (5 if i%17 else 60)  # jitter + occasional spike
        ok = (i % 17) != 0
        qd = max(0.0, 2.0*math.sin(i/13.0))
        al.record_sample(VEN, rtt_ms=rtt, ok=ok, q_delay_ms=qd)
        if i % 20 == 0:
            print({
                "timeout_ms": al.timeout_ms(VEN),
                "pace_ms": al.pacing_interval_ms(VEN),
                "burst": al.burst_limit(VEN),
                "open": al.is_open(VEN)
            })
        time.sleep(0.01)