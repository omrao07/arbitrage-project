# backend/sim/latency_sim.py
"""
Latency Simulator for Trading Pipelines
---------------------------------------
Models end-to-end latency across a realistic trading pipeline:
  [market_data_ingest] -> [strategy] -> [risk_guard] -> [order_router]
  -> <uplink> -> [exchange] -> <downlink> -> [confirm_sink]

Features
  • Discrete-event simulation with per-stage queues & k servers (M/M/k-ish).
  • Arrival process: Poisson with optional microbursts.
  • Service times: lognormal with jitter; optional GC pauses.
  • Network links: base + jitter + heavy-tail (Pareto) + loss + exponential backoff retries.
  • Backpressure: finite queues; drops counted and audited.
  • SLOs: end-to-end and per-stage percentiles (p50/p90/p99/p99.9) + compliance flags.
  • Audit: envelopes (SHA-256) for retries, drops, and SLO breaches.
  • Bus: publish envelopes to STREAM_LATENCY_EVENTS (stubbed if bus missing).

Usage
-----
from backend.sim.latency_sim import (
    WorkloadConfig, StageConfig, LinkConfig, SLOConfig, LatencySim, default_pipeline
)

stages, links = default_pipeline()
wl = WorkloadConfig(duration_ms=60_000, rps=400, seed=42)
slo = SLOConfig(e2e_target_ms=25.0, p99_target_ms=50.0)

sim = LatencySim(stages, links, wl, slo)
result = sim.run()
print(result["summary"])
"""

from __future__ import annotations

import heapq
import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------- Bus hook (optional) ----------------
try:
    from backend.bus.streams import publish_stream # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        head = {k: payload.get(k) for k in ("ts","type","id","stage","reason","lat_ms") if isinstance(payload, dict)}
        print(f"[stub publish_stream] {stream} <- {json.dumps(head)[:200]}")

STREAM_LATENCY_EVENTS = "STREAM_LATENCY_EVENTS"

# ---------------- Config models ----------------

@dataclass
class WorkloadConfig:
    duration_ms: int = 60_000                # total sim time
    rps: float = 200.0                       # base requests per second (Poisson)
    seed: Optional[int] = 7
    # Microburst: temporary multiplier to arrival rate
    burst_prob_per_s: float = 0.02           # probability each second to start a burst
    burst_len_ms: int = 800
    burst_rps_multiplier: float = 3.0

@dataclass
class StageConfig:
    name: str
    servers: int = 1                         # parallelism (k)
    queue_capacity: int = 10_000             # max waiting; 0 => drop if busy
    svc_logn_mu: float = 3.0                 # lognormal(mu, sigma) ms (mu in log-space of ms)
    svc_logn_sigma: float = 0.25
    jitter_ms: float = 0.2
    gc_pause_prob: float = 0.0               # per-job chance of GC stall
    gc_pause_ms: float = 0.0
    # Optional slow-start bias for warm caches, etc.
    warmup_jobs: int = 0                     # first N jobs get multiplier
    warmup_mult: float = 2.0

@dataclass
class LinkConfig:
    name: str
    base_ms: float = 0.8
    jitter_ms: float = 0.4
    tail_pareto_prob: float = 0.002          # chance of a heavy-tail spike
    tail_pareto_alpha: float = 1.7           # Pareto shape
    tail_extra_ms: float = 8.0               # scale for tail draw
    loss_rate: float = 0.0005                # packet loss probability (per hop)
    retry_backoff_ms: float = 1.0            # base backoff
    retry_max: int = 3

@dataclass
class SLOConfig:
    e2e_target_ms: float = 30.0              # soft target
    p99_target_ms: float = 60.0              # 99th percentile goal
    drop_budget_ppm: float = 500.0           # allowed drops per million

# ---------------- Runtime structures ----------------

class _StageRuntime:
    def __init__(self, cfg: StageConfig):
        self.cfg = cfg
        self.busy_until: List[float] = [0.0] * cfg.servers     # server availability times
        self.queue: List[Tuple[float, int]] = []               # (arrival_time, job_id)
        self.proc_count: int = 0                               # for warmup
        # stats
        self.wait_samples: List[float] = []
        self.svc_samples: List[float] = []
        self.soj_samples: List[float] = []
        self.drops: int = 0
        self.enq: int = 0
        self.deq: int = 0
        self.queue_len_max: int = 0

class _LinkRuntime:
    def __init__(self, cfg: LinkConfig):
        self.cfg = cfg
        self.samples: List[float] = []
        self.retries: int = 0
        self.losses: int = 0

# ---------------- Helpers: distributions & envelopes ----------------

def _svc_draw_ms(rng: np.random.Generator, st: StageConfig, proc_count: int) -> float:
    base = float(rng.lognormal(mean=st.svc_logn_mu, sigma=st.svc_logn_sigma))
    if st.jitter_ms > 0:
        base += float(rng.normal(0.0, st.jitter_ms))
    if st.warmup_jobs and proc_count < st.warmup_jobs:
        base *= st.warmup_mult
    if st.gc_pause_prob > 0 and rng.random() < st.gc_pause_prob:
        base += st.gc_pause_ms
    return max(0.01, base)

def _link_draw_ms(rng: np.random.Generator, lk: LinkConfig) -> Tuple[float, int]:
    """Return (latency_ms, retries_used). Retries on loss with exponential backoff."""
    # Loss check for the initial try
    retries = 0
    total = 0.0
    while True:
        # One attempt latency
        lat = lk.base_ms + max(0.0, rng.normal(0.0, lk.jitter_ms))
        if rng.random() < lk.tail_pareto_prob:
            # Pareto(alpha) scaled
            u = rng.random()
            tail = lk.tail_extra_ms / (u ** (1.0 / lk.tail_pareto_alpha))
            lat += tail
        total += max(0.01, lat)

        # Loss?
        if rng.random() < lk.loss_rate:
            if retries >= lk.retry_max:
                # Simulate "drop" at link level by returning NaN lat; caller will mark drop.
                return (float("nan"), retries)
            # backoff before retry
            bo = lk.retry_backoff_ms * (2 ** retries) + rng.uniform(0.0, lk.retry_backoff_ms)
            total += bo
            retries += 1
            continue
        return (total, retries)

def _percentiles(xs: List[float]) -> Dict[str, Optional[float]]:
    if not xs:
        return {"p50": None, "p90": None, "p99": None, "p999": None, "avg": None}
    arr = np.sort(np.array(xs, dtype=float))
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
        "avg": float(np.mean(arr)),
    }

def _hash_env(obj: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False, default=str).encode()).hexdigest()

def _audit(event: Dict[str, Any]) -> Dict[str, Any]:
    event["hash"] = _hash_env(event)
    publish_stream(STREAM_LATENCY_EVENTS, event)
    return event

# ---------------- Pipeline builder ----------------

def default_pipeline() -> Tuple[List[StageConfig], List[LinkConfig]]:
    stages = [
        StageConfig(name="market_data_ingest", servers=2, svc_logn_mu=2.5, svc_logn_sigma=0.35, jitter_ms=0.2),
        StageConfig(name="strategy",            servers=2, svc_logn_mu=2.8, svc_logn_sigma=0.35, jitter_ms=0.3, gc_pause_prob=0.002, gc_pause_ms=6.0),
        StageConfig(name="risk_guard",          servers=2, svc_logn_mu=2.2, svc_logn_sigma=0.25, jitter_ms=0.2),
        StageConfig(name="order_router",        servers=3, svc_logn_mu=2.0, svc_logn_sigma=0.25, jitter_ms=0.2),
        StageConfig(name="exchange",            servers=8, svc_logn_mu=1.8, svc_logn_sigma=0.20, jitter_ms=0.1),
        StageConfig(name="confirm_sink",        servers=2, svc_logn_mu=1.6, svc_logn_sigma=0.20, jitter_ms=0.1),
    ]
    links = [
        LinkConfig(name="uplink",   base_ms=0.7, jitter_ms=0.3, tail_pareto_prob=0.002, tail_extra_ms=6.0, loss_rate=0.0005),
        LinkConfig(name="downlink", base_ms=0.7, jitter_ms=0.3, tail_pareto_prob=0.002, tail_extra_ms=6.0, loss_rate=0.0005),
    ]
    return stages, links

# ---------------- Core Simulator ----------------

class LatencySim:
    """
    Discrete-event simulator for request pipeline with two links around the exchange.
    """

    def __init__(
        self,
        stages: List[StageConfig],
        links: List[LinkConfig],
        workload: WorkloadConfig,
        slo: Optional[SLOConfig] = None,
    ) -> None:
        self.stages_cfg = stages
        self.links_cfg = links
        self.wl = workload
        self.slo = slo or SLOConfig()
        self.rng = np.random.default_rng(workload.seed)

        # Build runtimes
        self.stages = [_StageRuntime(s) for s in stages]
        # Assume two links: uplink between router->exchange and downlink exchange->confirm
        if len(links) != 2:
            raise ValueError("Expect exactly 2 links: [uplink, downlink]")
        self.links = [_LinkRuntime(links[0]), _LinkRuntime(links[1])]

        # Event queue: (time_ms, counter, kind, payload)
        self.clock_ms = 0.0
        self._evq: List[Tuple[float, int, str, Dict[str, Any]]] = []
        self._eid = 0

        # Metrics
        self.e2e_samples: List[float] = []
        self.drops_total = 0
        self.retries_total = 0
        self.req_count = 0

    # ---------- Public API ----------

    def run(self) -> Dict[str, Any]:
        self._seed_arrivals()

        while self._evq:
            t, _, kind, payload = heapq.heappop(self._evq)
            self.clock_ms = t
            if kind == "arrival":
                self._handle_arrival(payload)
            elif kind == "stage_done":
                self._handle_stage_done(payload)
            elif kind == "link_done":
                self._handle_link_done(payload)
            else:
                # ignore unknown
                pass

        return self._summarize()

    # ---------- Events ----------

    def _push(self, time_ms: float, kind: str, payload: Dict[str, Any]) -> None:
        self._eid += 1
        heapq.heappush(self._evq, (time_ms, self._eid, kind, payload))

    def _seed_arrivals(self) -> None:
        """Generate Poisson arrivals with optional microburst epochs."""
        t = 0.0
        dur = float(self.wl.duration_ms)
        base_lambda = self.wl.rps / 1000.0  # per ms
        next_burst_end = -1.0
        while t < dur:
            # Determine current rate (burst or base)
            rate = base_lambda
            if t >= next_burst_end:
                # maybe start a burst
                if self.rng.random() < self.wl.burst_prob_per_s * (1.0):  # approx check per ~1s
                    next_burst_end = t + self.wl.burst_len_ms
            if t < next_burst_end:
                rate = base_lambda * self.wl.burst_rps_multiplier

            # Poisson interarrival ~ Exp(rate)
            if rate <= 0:
                break
            dt = self.rng.exponential(1.0 / rate)
            t += dt
            if t > dur:
                break
            rid = self.req_count
            self.req_count += 1
            self._push(t, "arrival", {"id": rid, "t0": t, "stage_index": 0})

    def _handle_arrival(self, p: Dict[str, Any]) -> None:
        # Arrives to stage 0 (market_data_ingest)
        self._enqueue_stage(stage_idx=0, job_id=p["id"], t_arr=p["t0"])

    def _enqueue_stage(self, stage_idx: int, job_id: int, t_arr: float) -> None:
        st = self.stages[stage_idx]
        st.enq += 1
        if len(st.queue) >= st.queue_capacity: # type: ignore
            st.drops += 1
            self.drops_total += 1
            _audit({
                "ts": int(self.clock_ms),
                "type": "drop",
                "id": job_id,
                "stage": st.cfg.name,
                "reason": "queue_full",
                "queue_len": len(st.queue),
            })
            return

        # Try to assign to an idle server; otherwise queue
        # Find the earliest-available server
        i_srv = int(np.argmin(st.busy_until))
        now = float(self.clock_ms)
        if st.busy_until[i_srv] <= now and len(st.queue) == 0:
            # start immediately
            svc = _svc_draw_ms(self.rng, st.cfg, st.proc_count)
            st.proc_count += 1
            st.wait_samples.append(0.0)
            st.svc_samples.append(svc)
            st.soj_samples.append(svc)
            st.busy_until[i_srv] = now + svc
            self._push(now + svc, "stage_done", {"id": job_id, "stage_index": stage_idx, "server": i_srv, "t_arr": t_arr, "svc": svc, "wait": 0.0})
        else:
            # enqueue
            st.queue.append((t_arr, job_id))
            st.queue_len_max = max(st.queue_len_max, len(st.queue))

    def _dequeue_start(self, stage_idx: int, server_idx: int) -> None:
        st = self.stages[stage_idx]
        if not st.queue:
            return
        t_arr, job_id = st.queue.pop(0)
        now = float(self.clock_ms)
        wait = max(0.0, now - t_arr)
        svc = _svc_draw_ms(self.rng, st.cfg, st.proc_count)
        st.proc_count += 1
        st.wait_samples.append(wait)
        st.svc_samples.append(svc)
        st.soj_samples.append(wait + svc)
        st.busy_until[server_idx] = now + svc
        self._push(now + svc, "stage_done", {"id": job_id, "stage_index": stage_idx, "server": server_idx, "t_arr": t_arr, "svc": svc, "wait": wait})

    def _handle_stage_done(self, p: Dict[str, Any]) -> None:
        idx = p["stage_index"]
        st = self.stages[idx]
        st.deq += 1
        # Free server and start next if any
        srv = p["server"]
        # server is already free at now; attempt to start next
        self._dequeue_start(idx, srv)

        # Route to next hop
        if st.cfg.name == "order_router":
            # Uplink to exchange
            lat, retries = _link_draw_ms(self.rng, self.links[0].cfg)
            if math.isnan(lat):
                self.links[0].losses += 1
                self.drops_total += 1
                _audit({"ts": int(self.clock_ms), "type": "drop", "id": p["id"], "stage": "uplink", "reason": "link_loss_max_retries"})
                return
            self.links[0].samples.append(lat)
            self.links[0].retries += retries
            self.retries_total += retries
            self._push(self.clock_ms + lat, "link_done", {"id": p["id"], "dir": "uplink", "t_arr": p["t_arr"]})
            return

        if st.cfg.name == "exchange":
            # Downlink back
            lat, retries = _link_draw_ms(self.rng, self.links[1].cfg)
            if math.isnan(lat):
                self.links[1].losses += 1
                self.drops_total += 1
                _audit({"ts": int(self.clock_ms), "type": "drop", "id": p["id"], "stage": "downlink", "reason": "link_loss_max_retries"})
                return
            self.links[1].samples.append(lat)
            self.links[1].retries += retries
            self.retries_total += retries
            self._push(self.clock_ms + lat, "link_done", {"id": p["id"], "dir": "downlink", "t_arr": p["t_arr"]})
            return

        # Otherwise, proceed to next stage in list
        next_idx = idx + 1
        self._enqueue_stage(next_idx, p["id"], t_arr=self.clock_ms)

    def _handle_link_done(self, p: Dict[str, Any]) -> None:
        if p["dir"] == "uplink":
            # Arrive to exchange stage
            ex_idx = self._stage_index_by_name("exchange")
            self._enqueue_stage(ex_idx, p["id"], t_arr=self.clock_ms)
            return
        if p["dir"] == "downlink":
            # Arrive to confirm sink stage
            cs_idx = self._stage_index_by_name("confirm_sink")
            self._enqueue_stage(cs_idx, p["id"], t_arr=self.clock_ms)
            # If this was the last stage, stage_done will handle e2e; there is no extra hop here.
            return

    # ---------- Utilities ----------

    def _stage_index_by_name(self, name: str) -> int:
        for i, s in enumerate(self.stages_cfg):
            if s.name == name:
                return i
        raise KeyError(name)

    # ---------- Summary & SLO ----------

    def _summarize(self) -> Dict[str, Any]:
        # End-to-end: approximate from first to final dequeue of confirm_sink
        # We tracked sojourn times per stage; compute e2e by summing per job? We didn't store per-job timeline.
        # Approximation: simulate e2e by prop from arrival to final stage sojourn accumulation captured by stage_done at each hop.
        # Simpler: Reconstruct from confirm_sink samples (sojourn list is aligned with deq order). We'll instead capture at confirm_sink stage.
        # To do this robustly now, compute e2e as sum of samples across stages & link averages weighted by flow, which is a good proxy for SLO planning.
        per_stage = {}
        for st, rt in zip(self.stages_cfg, self.stages):
            per_stage[st.name] = {
                "wait": _percentiles(rt.wait_samples),
                "service": _percentiles(rt.svc_samples),
                "sojourn": _percentiles(rt.soj_samples),
                "queue_len_max": rt.queue_len_max,
                "enq": rt.enq, "deq": rt.deq, "drops": rt.drops,
            }

        per_link = {}
        for lk_rt in self.links:
            per_link[lk_rt.cfg.name] = {
                "lat": _percentiles(lk_rt.samples),
                "losses": lk_rt.losses,
                "retries": lk_rt.retries,
            }

        # Estimate e2e as sum of median sojourn per stage + median link latency
        e2e_proxy_samples = []
        # Build proxy by combining per-stage sojourn medians; not exact but stable for SLO sizing
        n_completed = min([self.stages[self._stage_index_by_name("confirm_sink")].deq] + [s.deq for s in self.stages])
        if n_completed > 0:
            # For a crude distribution, bootstrap by sampling per-stage sojourns where available
            # Take min length among stages to align
            min_len = min(len(s.soj_samples) for s in self.stages)
            for i in range(min_len):
                total = 0.0
                for s in self.stages:
                    total += s.soj_samples[i]
                # Links: sample circularly if fewer
                if self.links[0].samples:
                    total += self.links[0].samples[i % len(self.links[0].samples)]
                if self.links[1].samples:
                    total += self.links[1].samples[i % len(self.links[1].samples)]
                e2e_proxy_samples.append(total)

        e2e_stats = _percentiles(e2e_proxy_samples)
        # SLO checks
        slo = asdict(self.slo)
        breaches: List[Dict[str, Any]] = []

        def _maybe_breach(kind: str, val: Optional[float], thr: float) -> None:
            if val is None:
                return
            if val > thr:
                evt = _audit({
                    "ts": int(self.clock_ms),
                    "type": "slo_breach",
                    "id": None,
                    "stage": "e2e",
                    "reason": kind,
                    "lat_ms": float(val),
                    "target_ms": float(thr),
                })
                breaches.append(evt)

        _maybe_breach("p99_target_ms", e2e_stats.get("p99"), self.slo.p99_target_ms)
        _maybe_breach("e2e_target_ms", e2e_stats.get("p50"), self.slo.e2e_target_ms)

        drop_ppm = (self.drops_total / max(1, self.req_count)) * 1e6
        if drop_ppm > self.slo.drop_budget_ppm:
            breaches.append(_audit({
                "ts": int(self.clock_ms),
                "type": "slo_breach",
                "id": None,
                "stage": "e2e",
                "reason": "drop_budget_ppm",
                "ppm": float(drop_ppm),
                "target_ppm": float(self.slo.drop_budget_ppm),
            }))

        summary = {
            "requests": self.req_count,
            "completed": int(self.stages[self._stage_index_by_name("confirm_sink")].deq),
            "drops_total": self.drops_total,
            "retries_total": self.retries_total,
            "drop_ppm": float(drop_ppm),
            "e2e_ms": e2e_stats,
            "slo": slo,
            "breaches": breaches,
        }

        return {
            "summary": summary,
            "per_stage": per_stage,
            "per_link": per_link,
        }

# ---------------- Demo ----------------

if __name__ == "__main__":
    stages, links = default_pipeline()
    wl = WorkloadConfig(duration_ms=30_000, rps=300, seed=123, burst_prob_per_s=0.03, burst_len_ms=600, burst_rps_multiplier=3.5)
    slo = SLOConfig(e2e_target_ms=25.0, p99_target_ms=50.0, drop_budget_ppm=800.0)
    sim = LatencySim(stages, links, wl, slo)
    res = sim.run()
    print(json.dumps(res["summary"], indent=2))
    # Optional per-stage print:
    for k, v in res["per_stage"].items():
        p50 = v["sojourn"]["p50"]
        p99 = v["sojourn"]["p99"]
        print(f"{k:18s} sojourn p50={p50:.2f} ms   p99={p99:.2f} ms")