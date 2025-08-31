# backend/ai/ensemble/ensemble_switcher.py
from __future__ import annotations

"""
Ensemble Switcher
-----------------
Policy-driven selector for models/strategies:
 - Supports: "rules", "epsilon_greedy", "softmax", "ucb1", "thompson"
 - Blends online performance metrics with risk/regime constraints
 - Cooldowns & min-hold to reduce thrash
 - Optional Redis emit; JSON snapshot for dashboards

Typical usage
-------------
sw = EnsembleSwitcher(
    policy="softmax",
    epsilon=0.05,
    temperature=0.6,
    min_hold_s=30,
    cooldown_s=10,
    risk_max_drawdown=0.12,
    risk_max_vol=0.35
)

sw.register("meanrev_v1", pri=1.0, tags={"us","intraday"})
sw.register("mom_v2",     pri=1.0, tags={"us","swing"})
sw.register("macro_gbt",  pri=0.8, tags={"global","macro"})

# feed metrics periodically (rolling):
sw.update_metrics("meanrev_v1", sharpe=1.1, hit=0.57, ann_vol=0.18, dd=0.05, latency_ms=2.5)
sw.update_metrics("mom_v2",     sharpe=0.8, hit=0.52, ann_vol=0.14, dd=0.04, latency_ms=3.1)
sw.update_metrics("macro_gbt",  sharpe=0.5, hit=0.51, ann_vol=0.10, dd=0.03, latency_ms=8.2)

# make a decision given context (e.g., regime tags)
dec = sw.decide(context={"regime": ["Goldilocks", "Risk-On"], "region": "US"})
print(dec.chosen, dec.weights)
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import numpy as _np
except Exception:
    _np = None

# ---------- Optional Redis bus (graceful fallback) ----------
try:
    import redis as _redis
except Exception:
    _redis = None

try:
    from backend.bus.streams import publish_stream  # type: ignore
except Exception:
    def publish_stream(stream: str, payload: Dict[str, Any]) -> None:
        pass

ENSM_OUT_STREAM = os.getenv("ENSEMBLE_OUT_STREAM", "ensemble.decisions")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

# ----------------------- Data Models -----------------------

@dataclass
class Candidate:
    name: str
    pri: float = 1.0                   # prior preference / static weight
    tags: Set[str] = field(default_factory=set)
    enabled: bool = True
    last_pick_ts: float = 0.0
    picks: int = 0
    wins: int = 0
    losses: int = 0

@dataclass
class Metrics:
    sharpe: float = 0.0
    hit: float = 0.5                   # hit rate (0..1)
    ann_vol: float = 0.0               # annualized vol (as fraction)
    dd: float = 0.0                    # max drawdown (fraction)
    lat_ms: float = 0.0                # execution latency
    pnl_per_trade: float = 0.0         # optional
    reward: float = 0.0                # smoothed reward used by bandits
    ts: float = 0.0

@dataclass
class Decision:
    ts: int
    chosen: str
    weights: Dict[str, float]          # allocation across candidates (can be one-hot)
    policy: str
    reason: str
    context: Dict[str, Any]

# ----------------------- Switcher Core -----------------------

class EnsembleSwitcher:
    def __init__(
        self,
        *,
        policy: str = "softmax",               # "rules"|"epsilon_greedy"|"softmax"|"ucb1"|"thompson"
        epsilon: float = 0.05,
        temperature: float = 0.7,
        ucb_c: float = 1.2,
        min_hold_s: float = 20.0,
        cooldown_s: float = 10.0,
        risk_max_drawdown: float = 0.2,
        risk_max_vol: float = 0.5,
        prefer_tags: Optional[Set[str]] = None,   # boost candidates with these tags in context regime
        out_stream: str = ENSM_OUT_STREAM
    ):
        self.policy = policy
        self.epsilon = epsilon
        self.temperature = max(1e-3, temperature)
        self.ucb_c = ucb_c
        self.min_hold_s = max(0.0, min_hold_s)
        self.cooldown_s = max(0.0, cooldown_s)
        self.risk_max_drawdown = risk_max_drawdown
        self.risk_max_vol = risk_max_vol
        self.prefer_tags = set(prefer_tags or set())
        self.out_stream = out_stream

        self.c: Dict[str, Candidate] = {}
        self.m: Dict[str, Metrics] = {}
        self.current: Optional[str] = None
        self.current_ts: float = 0.0
        self._r = None
        if _redis is not None:
            try:
                self._r = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
            except Exception:
                self._r = None

        # bandit priors for thompson (Beta for hit-rate)
        self._alpha: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}

    # --------------- Registry & Metrics ----------------

    def register(self, name: str, *, pri: float = 1.0, tags: Optional[Set[str]] = None, enabled: bool = True) -> None:
        self.c[name] = Candidate(name=name, pri=float(pri), tags=set(tags or set()), enabled=enabled)
        if name not in self.m:
            self.m[name] = Metrics()
        self._alpha.setdefault(name, 1.0)
        self._beta.setdefault(name, 1.0)

    def enable(self, name: str, flag: bool = True) -> None:
        if name in self.c:
            self.c[name].enabled = flag

    def update_metrics(
        self,
        name: str,
        *,
        sharpe: Optional[float] = None,
        hit: Optional[float] = None,
        ann_vol: Optional[float] = None,
        dd: Optional[float] = None,
        latency_ms: Optional[float] = None,
        pnl_per_trade: Optional[float] = None,
        reward: Optional[float] = None,
        smooth: float = 0.5
    ) -> None:
        """
        Update rolling metrics for a candidate. 'reward' is the scalar used by bandit policies.
        If reward is None, derive from sharpe/hit/latency with a sensible heuristic.
        """
        if name not in self.m:
            self.m[name] = Metrics()
        mt = self.m[name]
        now = time.time()
        def _ema(cur, new):
            return (1.0 - smooth) * float(cur) + smooth * float(new)

        if sharpe is not None: mt.sharpe = _ema(mt.sharpe, sharpe)
        if hit is not None:    mt.hit = _ema(mt.hit, hit)
        if ann_vol is not None: mt.ann_vol = _ema(mt.ann_vol, ann_vol)
        if dd is not None:     mt.dd = _ema(mt.dd, dd)
        if latency_ms is not None: mt.lat_ms = _ema(mt.lat_ms, latency_ms)
        if pnl_per_trade is not None: mt.pnl_per_trade = _ema(mt.pnl_per_trade, pnl_per_trade)

        # derive reward if not provided: higher sharpe/hit, lower latency/vol/dd
        if reward is None:
            # normalize components to ~[-1,1]
            r = 0.0
            r += 0.6 * math.tanh((mt.sharpe or 0.0) / 1.0)
            r += 0.3 * (2.0 * (mt.hit or 0.5) - 1.0)
            r -= 0.2 * math.tanh((mt.ann_vol or 0.0) / 0.4)
            r -= 0.2 * math.tanh((mt.dd or 0.0) / 0.2)
            r -= 0.1 * math.tanh((mt.lat_ms or 0.0) / 10.0)
            reward = r
        mt.reward = _ema(mt.reward, reward)
        mt.ts = now

    def record_outcome(self, name: str, *, win: bool, amount: float = 1.0) -> None:
        if name not in self.c:
            return
        cand = self.c[name]
        cand.picks += 1
        if win:
            cand.wins += 1
            self._alpha[name] = self._alpha.get(name, 1.0) + amount
        else:
            cand.losses += 1
            self._beta[name] = self._beta.get(name, 1.0) + amount

    # --------------- Decision Logic -------------------

    def decide(self, *, context: Optional[Dict[str, Any]] = None, k: int = 1) -> Decision:
        """
        Returns a Decision with a chosen candidate and (optional) allocation weights.
        If k>1, returns a soft allocation across top-k by policy.
        """
        ctx = context or {}
        now = time.time()
        avail = self._eligible(ctx, now)
        if not avail:
            # fallback: if everything filtered, allow enabled regardless of risk
            avail = [n for n, c in self.c.items() if c.enabled]

        if not avail:
            raise RuntimeError("No available candidates to choose from")

        scores = self._scores(avail, ctx, now)
        order = sorted(avail, key=lambda n: scores.get(n, -1e9), reverse=True)

        # selection by policy
        if self.policy == "rules":
            chosen, weights, reason = self._rules_pick(order, scores, k)
        elif self.policy == "epsilon_greedy":
            chosen, weights, reason = self._eps_pick(order, scores, k)
        elif self.policy == "softmax":
            chosen, weights, reason = self._softmax_pick(avail, scores, k)
        elif self.policy == "ucb1":
            chosen, weights, reason = self._ucb_pick(avail, scores, k)
        elif self.policy == "thompson":
            chosen, weights, reason = self._thompson_pick(avail, scores, k)
        else:
            raise ValueError(f"Unknown policy: {self.policy}")

        # min-hold / cooldown enforcement (stickiness)
        if self.current and (now - self.current_ts) < self.min_hold_s:
            chosen = self.current
            weights = {self.current: 1.0}
            reason = f"hold({self.min_hold_s:.0f}s)"
        else:
            self.current = chosen
            self.current_ts = now
            self.c[chosen].last_pick_ts = now

        dec = Decision(
            ts=int(now * 1000),
            chosen=chosen,
            weights=weights,
            policy=self.policy,
            reason=reason,
            context=ctx
        )
        self._emit(dec)
        return dec

    # ---------------- Internal Helpers ----------------

    def _eligible(self, ctx: Dict[str, Any], now: float) -> List[str]:
        out = []
        regime_tags = set([t.lower() for t in ctx.get("regime", [])])
        for n, cand in self.c.items():
            if not cand.enabled:
                continue
            met = self.m.get(n, Metrics())
            # risk gates
            if (met.dd or 0.0) > self.risk_max_drawdown:
                continue
            if (met.ann_vol or 0.0) > self.risk_max_vol:
                continue
            # cooldown
            if (now - (cand.last_pick_ts or 0.0)) < self.cooldown_s and n != self.current:
                continue
            # regime preference (soft boost handled in scoring)
            out.append(n)
        return out

    def _scores(self, avail: List[str], ctx: Dict[str, Any], now: float) -> Dict[str, float]:
        """
        Base score uses Metrics.reward + pri + tag boosts + recency decay.
        """
        scores: Dict[str, float] = {}
        regime_tags = set([t.lower() for t in ctx.get("regime", [])])
        region = (ctx.get("region") or "").lower()

        for n in avail:
            cand = self.c[n]
            mt = self.m.get(n, Metrics())
            base = mt.reward + 0.1 * math.tanh(cand.pri)
            # tag boost if candidate tags overlap regime/region
            overlap = len(cand.tags & (regime_tags | ({region} if region else set())))
            base += 0.05 * overlap
            # slight recency penalty to encourage exploration
            age = now - (cand.last_pick_ts or 0.0)
            base += 0.01 * math.tanh(age / max(1.0, self.min_hold_s))
            scores[n] = base
        return scores

    def _rules_pick(self, ordered: List[str], scores: Dict[str, float], k: int) -> Tuple[str, Dict[str, float], str]:
        best = ordered[0]
        if k <= 1:
            return best, {best: 1.0}, "rules_top1"
        # simple normalized weights from scores for top-k
        top = ordered[:k]
        vals = [max(1e-9, scores[t]) for t in top]
        s = sum(vals)
        w = {t: v / s for t, v in zip(top, vals)}
        return top[0], w, "rules_topk"

    def _eps_pick(self, ordered: List[str], scores: Dict[str, float], k: int) -> Tuple[str, Dict[str, float], str]:
        if random.random() < self.epsilon:
            rnd = random.choice(ordered)
            return rnd, {rnd: 1.0}, f"eps_explore({self.epsilon:.2f})"
        return self._rules_pick(ordered, scores, k)

    def _softmax_pick(self, avail: List[str], scores: Dict[str, float], k: int) -> Tuple[str, Dict[str, float], str]:
        if _np is None:
            # fallback to epsilon-greedy if numpy missing
            ordered = sorted(avail, key=lambda n: scores[n], reverse=True)
            return self._eps_pick(ordered, scores, k)
        x = _np.array([scores[n] for n in avail], dtype=float)
        x = x - x.max()  # stabilize
        p = _np.exp(x / self.temperature)
        p = p / p.sum()
        if k <= 1:
            idx = int(_np.random.choice(len(avail), p=p))
            name = avail[idx]
            return name, {name: 1.0}, "softmax"
        # sample k without replacement proportional to p, then normalize weights by p
        idxs = list(range(len(avail)))
        chosen_idx = _np.random.choice(idxs, size=min(k, len(idxs)), replace=False, p=p)
        weights = {avail[i]: float(p[i]) for i in chosen_idx}
        z = sum(weights.values())
        weights = {k_: v / z for k_, v in weights.items()}
        lead = max(weights.items(), key=lambda kv: kv[1])[0]
        return lead, weights, "softmax_topk"

    def _ucb_pick(self, avail: List[str], scores: Dict[str, float], k: int) -> Tuple[str, Dict[str, float], str]:
        t = 1 + sum(self.c[n].picks for n in avail)
        vals: List[Tuple[str, float]] = []
        for n in avail:
            c = self.c[n]
            mt = self.m.get(n, Metrics())
            base = mt.reward
            n_pick = max(1, c.picks)
            bonus = self.ucb_c * math.sqrt(math.log(t) / n_pick)
            vals.append((n, base + bonus))
        vals.sort(key=lambda kv: kv[1], reverse=True)
        if k <= 1:
            return vals[0][0], {vals[0][0]: 1.0}, "ucb1"
        top = vals[:k]
        s = sum(max(1e-9, v) for _, v in top)
        w = {n: max(1e-9, v) / s for n, v in top}
        lead = top[0][0]
        return lead, w, "ucb1_topk"

    def _thompson_pick(self, avail: List[str], scores: Dict[str, float], k: int) -> Tuple[str, Dict[str, float], str]:
        draws: List[Tuple[str, float]] = []
        for n in avail:
            a = self._alpha.get(n, 1.0)
            b = self._beta.get(n, 1.0)
            # simple Beta draw on hit-rate; blend reward as mean shift
            if _np is not None:
                x = _np.random.beta(a, b)
            else:
                # poor man's beta via mean of uniforms
                x = sum(random.random() for _ in range(6)) / 6.0
            x = 0.7 * x + 0.3 * (scores[n] * 0.5 + 0.5)  # map reward ~[-1,1] -> [0,1]
            draws.append((n, x))
        draws.sort(key=lambda kv: kv[1], reverse=True)
        if k <= 1:
            return draws[0][0], {draws[0][0]: 1.0}, "thompson"
        top = draws[:k]
        s = sum(v for _, v in top)
        w = {n: v / s for n, v in top}
        lead = top[0][0]
        return lead, w, "thompson_topk"

    # --------------- Emission & Snapshot ----------------

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ts": int(time.time() * 1000),
            "policy": self.policy,
            "current": self.current,
            "candidates": {n: asdict(c) for n, c in self.c.items()},
            "metrics": {n: asdict(m) for n, m in self.m.items()},
            "bandit": {"alpha": self._alpha, "beta": self._beta}
        }

    def _emit(self, dec: Decision) -> None:
        payload = {
            "decision": asdict(dec),
            "state": {"current": self.current}
        }
        publish_stream(self.out_stream, payload)
        # Optionally store snapshot in Redis for UI panels
        if self._r is not None:
            try:
                self._r.set("ensemble:snapshot", json.dumps(self.snapshot()))
            except Exception:
                pass