# backend/risk/adversary.py
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Protocol, runtime_checkable

# ----------------------------- Soft types ------------------------------------
try:
    from backend.execution.pricer import Quote # type: ignore
except Exception:
    @dataclass
    class Quote:
        symbol: str
        bid: Optional[float] = None
        ask: Optional[float] = None
        last: Optional[float] = None
        ts: float = field(default_factory=lambda: time.time())
        def mid(self) -> Optional[float]:
            if self.bid and self.ask:
                return (self.bid + self.ask) / 2.0
            return self.last

# Minimal trade/order mirrors to avoid hard deps
@dataclass
class Order:
    symbol: str
    side: str     # 'buy' | 'sell'
    qty: float
    tif: str = "IOC"
    limit_price: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Trade:
    symbol: str
    price: float
    qty: float
    side: str            # aggressor side
    venue: str
    ts: float
    fees: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

# ----------------------------- Interfaces ------------------------------------

@runtime_checkable
class Adversary(Protocol):
    """
    Strategy-agnostic adversary interface:
      - mutate NBBO
      - front-run/drag fills
      - inject spoof/cancel patterns
      - compute a toxicity score for pre-trade guardrails
    """
    name: str
    def step_nbbo(self, q: Quote) -> Quote: ...
    def before_send(self, o: Order, nbbo: Quote) -> Order: ...
    def after_fill(self, t: Trade, nbbo: Quote) -> Trade: ...
    def toxicity(self, o: Order, nbbo: Quote) -> float: ...

# ----------------------------- Adversaries -----------------------------------

@dataclass
class Spoofer:
    """
    Adversary that widens spreads by flashing/lifting layered quotes.
    Effects:
      - transiently widens NBBO by 'widen_bps' with half-life 'hl_ms'
      - increases taker slippage when crossing the spread
    """
    name: str = "spoofer"
    widen_bps: float = 8.0         # temporary spread widen (bps of mid)
    slippage_bps: float = 2.0      # extra taker slippage (bps of mid)
    hl_ms: int = 250               # half-life of spoof effect
    _last_spoof: Dict[str, float] = field(default_factory=dict)

    def _decay(self, sym: str) -> float:
        t0 = self._last_spoof.get(sym, 0.0)
        if t0 == 0.0: return 0.0
        dt = (time.time() - t0) * 1000.0
        # exponential decay in [0,1]
        return 2 ** (-dt / max(1.0, self.hl_ms))

    def step_nbbo(self, q: Quote) -> Quote:
        m = q.mid()
        if not m or not q.bid or not q.ask: return q
        # randomly spoof ~20% of ticks
        if random.random() < 0.2:
            self._last_spoof[q.symbol] = time.time()
        w = self._decay(q.symbol) * (self.widen_bps / 1e4) * m
        if w <= 0: return q
        return Quote(symbol=q.symbol, bid=q.bid - w/2, ask=q.ask + w/2, last=q.last, ts=q.ts)

    def before_send(self, o: Order, nbbo: Quote) -> Order:
        # no order mutation; spoofer acts via quotes/impact
        return o

    def after_fill(self, t: Trade, nbbo: Quote) -> Trade:
        m = nbbo.mid()
        if not m: return t
        # add extra taker slippage on aggressive fills (relative to mid)
        slip = (self.slippage_bps / 1e4) * m
        if t.side == "buy":
            t.price += slip
        else:
            t.price -= slip
        t.meta[self.name] = {"slip": slip}
        return t

    def toxicity(self, o: Order, nbbo: Quote) -> float:
        # toxic when spread recently widened and we cross
        m = nbbo.mid()
        if not m or not nbbo.bid or not nbbo.ask: return 0.0
        widen_now = (nbbo.ask - nbbo.bid) / m * 1e4
        tox = 0.0
        if o.tif in ("IOC","FOK") and ((o.side=="buy" and o.limit_price is None) or (o.side=="sell" and o.limit_price is None)):
            tox += min(1.0, max(0.0, (widen_now / max(1.0, self.widen_bps))))
        tox += 0.25 * self._decay(o.symbol)
        return min(1.0, tox)

@dataclass
class MomentumIgniter:
    """
    Adversary that 'runs ahead' of your aggressive order, shifting mid a few bps.
    """
    name: str = "momo"
    lead_bps: float = 6.0      # permanent mid shift on your aggression
    after_ms: int = 30         # delay before shift visible
    prob: float = 0.4

    def step_nbbo(self, q: Quote) -> Quote:
        # passive effect: none
        return q

    def before_send(self, o: Order, nbbo: Quote) -> Order:
        # no mutation
        return o

    def after_fill(self, t: Trade, nbbo: Quote) -> Trade:
        if random.random() > self.prob: return t
        m = nbbo.mid()
        if not m: return t
        shift = (self.lead_bps / 1e4) * m * (1 if t.side=="buy" else -1)
        # reflect new reference in trade meta; your pricer should pick this next tick
        t.meta[self.name] = {"mid_shift_after_ms": self.after_ms, "shift": shift}
        return t

    def toxicity(self, o: Order, nbbo: Quote) -> float:
        # aggressive marketables are riskier
        return 0.5 if (o.tif in ("IOC","FOK") and o.limit_price is None) else 0.1

@dataclass
class LatencySniper:
    """
    Adversary that picks off stale quotes; increases effective latency.
    """
    name: str = "sniper"
    add_latency_ms: int = 12       # added one-way latency
    extra_slip_bps: float = 1.2    # if quote stales beyond threshold
    stale_ms: int = 25

    def step_nbbo(self, q: Quote) -> Quote:
        # no NBBO change; acts via latency/after_fill
        return q

    def before_send(self, o: Order, nbbo: Quote) -> Order:
        # mark added latency so your router can simulate wall-clock delay
        o.meta[self.name] = {"added_ms": self.add_latency_ms}
        return o

    def after_fill(self, t: Trade, nbbo: Quote) -> Trade:
        m = nbbo.mid()
        if not m: return t
        age_ms = (time.time() - getattr(nbbo, "ts", time.time())) * 1000.0
        if age_ms > self.stale_ms:
            slip = (self.extra_slip_bps / 1e4) * m
            if t.side == "buy": t.price += slip
            else: t.price -= slip
            t.meta[self.name] = {"stale_ms": age_ms, "slip": slip}
        return t

    def toxicity(self, o: Order, nbbo: Quote) -> float:
        return min(1.0, max(0.0, (self.add_latency_ms / 50.0)))

# ----------------------------- Aggregator ------------------------------------

@dataclass
class AdversaryConfig:
    """
    Global knobs & gating.
    """
    enabled: bool = True
    max_toxicity: float = 0.7      # if predicted toxicity > this, advise NOT to cross
    slip_cap_bps: float = 25.0     # cap sum of adversary slippages
    random_seed: Optional[int] = None

class AdversarySuite:
    """
    Composes multiple adversaries and provides a simple API for routers/backtests.
    """
    def __init__(self, advs: List[Adversary], cfg: Optional[AdversaryConfig] = None):
        self.cfg = cfg or AdversaryConfig()
        self.advs = list(advs)
        if self.cfg.random_seed is not None:
            random.seed(self.cfg.random_seed)

    # --- pipeline hooks ---
    def mutate_nbbo(self, q: Quote) -> Quote:
        if not self.cfg.enabled: return q
        for a in self.advs:
            q = a.step_nbbo(q)
        return q

    def pre_trade_check(self, o: Order, nbbo: Quote) -> Dict[str, Any]:
        """
        Returns a dict with toxicity, advice, and per-adversary breakdown.
        Routers can use this for: do-not-cross / switch-to-passive / clip-size.
        """
        if not self.cfg.enabled:
            return {"toxicity": 0.0, "advice": "ok", "details": {}}

        scores: Dict[str, float] = {}
        for a in self.advs:
            try:
                scores[a.name] = float(max(0.0, min(1.0, a.toxicity(o, nbbo))))
            except Exception:
                scores[a.name] = 0.0
        tox = sum(scores.values()) / max(1, len(scores))
        advice = "avoid_cross" if tox >= self.cfg.max_toxicity else "ok"
        return {"toxicity": tox, "advice": advice, "details": scores}

    def before_send(self, o: Order, nbbo: Quote) -> Order:
        if not self.cfg.enabled: return o
        for a in self.advs:
            o = a.before_send(o, nbbo)
        return o

    def after_fill(self, t: Trade, nbbo: Quote) -> Trade:
        if not self.cfg.enabled: return t
        total_slip = 0.0
        for a in self.advs:
            t = a.after_fill(t, nbbo)
            # accumulate implied slippage if tagged
            meta = t.meta.get(a.name, {})
            total_slip += abs(float(meta.get("slip", 0.0)))
            if total_slip > (self.cfg.slip_cap_bps / 1e4) * (nbbo.mid() or t.price or 0):
                break
        return t

# ----------------------------- Guards / Policies ------------------------------

@dataclass
class GuardrailPolicy:
    """
    Simple policy that converts toxicity → execution choice.
    """
    avoid_cross_threshold: float = 0.7
    switch_to_passive_threshold: float = 0.5
    clip_scale_at_high_tox: float = 0.5

    def decide(self, precheck: Dict[str, Any], *, default: str = "CROSS") -> Dict[str, Any]:
        tox = float(precheck.get("toxicity", 0.0))
        if tox >= self.avoid_cross_threshold:
            return {"mode": "SKIP", "reason": "toxicity_high", "tox": tox}
        if tox >= self.switch_to_passive_threshold:
            return {"mode": "PASSIVE", "clip_scale": self.clip_scale_at_high_tox, "tox": tox}
        return {"mode": default, "tox": tox}

# ----------------------------- Quick wiring ----------------------------------

def default_suite(seed: Optional[int] = 7) -> AdversarySuite:
    """
    Sensible defaults: Spoofer + MomentumIgniter + LatencySniper.
    """
    advs: List[Adversary] = [
        Spoofer(widen_bps=8.0, slippage_bps=2.0, hl_ms=250),
        MomentumIgniter(lead_bps=6.0, after_ms=30, prob=0.4),
        LatencySniper(add_latency_ms=12, extra_slip_bps=1.2, stale_ms=25),
    ]
    return AdversarySuite(advs, AdversaryConfig(enabled=True, max_toxicity=0.7, slip_cap_bps=25.0, random_seed=seed))

# ----------------------------- Tiny demo -------------------------------------

if __name__ == "__main__":
    # Example usage with a fake NBBO and order
    suite = default_suite(seed=42)
    nbbo = Quote(symbol="AAPL", bid=192.00, ask=192.04, last=192.02)
    nbbo2 = suite.mutate_nbbo(nbbo)

    o = Order(symbol="AAPL", side="buy", qty=1000, tif="IOC")  # marketable
    pre = suite.pre_trade_check(o, nbbo2)
    policy = GuardrailPolicy()
    decision = policy.decide(pre, default="CROSS")
    print("Pre-check:", pre, "Decision:", decision)

    # Suppose we crossed and got a fill:
    t = Trade(symbol="AAPL", price=nbbo2.ask, qty=1000, side="buy", venue="SIM", ts=time.time()) # type: ignore
    t2 = suite.after_fill(t, nbbo2)
    print("Post-fill meta:", t2.meta)