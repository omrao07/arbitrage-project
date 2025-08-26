# backend/ai/alpha_swarm/genomes.py
"""
Genome schema for micro-alpha agents (the "alpha swarm").

Goals
-----
- Express small strategy blueprints as immutable-ish "genomes".
- Sample new genomes, mutate, crossover, score via external fitness.
- Compile genome -> runtime config callable (your engine decides how to wire).

No heavy deps; pure stdlib.

Typical flow
------------
from backend.ai.alpha_swarm.genomes import (
    StrategyGenome, ParameterSpace, Sampler, Mutator, Crossover
)

space = ParameterSpace.default()
sampler = Sampler(space)
mutator = Mutator(space)
xover   = Crossover(space)

g1 = sampler.sample(kind="momentum")
g2 = sampler.sample(kind="meanrev")

child = xover.crossover(g1, g2)
mut   = mutator.mutate(child, rate=0.2)

# Serialize
blob = mut.to_dict()
g3 = StrategyGenome.from_dict(blob)

# Compile into a tiny config that StrategyFactory can consume
cfg = g3.compile(symbol_universe=["AAPL","MSFT","RELIANCE.NS"])

Your swarm_manager can:
- keep a population of genomes
- request live fitness via your backtester/paper engine
- retire bad genomes, spawn from good ones
"""

from __future__ import annotations

import math
import random
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Iterable

# ------------------------------- Utilities -----------------------------------

def _utc_ms() -> int:
    return int(time.time() * 1000)

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _rand_log_uniform(lo: float, hi: float, rng: random.Random) -> float:
    """Sample over orders of magnitude (e.g., lookbacks)."""
    assert lo > 0 and hi > lo
    a, b = math.log(lo), math.log(hi)
    return math.exp(rng.uniform(a, b))

def _hash_dict(d: Dict[str, Any]) -> str:
    return hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest()[:16]

# ------------------------------- Spaces --------------------------------------

@dataclass(frozen=True)
class ParameterSpace:
    """
    Defines permissible ranges per strategy 'kind'.
    """
    momentum_lookback: Tuple[int, int] = (5, 120)          # bars
    momentum_exit_lb: Tuple[int, int] = (3, 60)
    meanrev_lb: Tuple[int, int] = (5, 80)
    meanrev_band_bps: Tuple[float, float] = (5.0, 80.0)    # entry threshold
    breakout_lb: Tuple[int, int] = (10, 200)
    breakout_confirm: Tuple[int, int] = (1, 5)
    pairs_half_life: Tuple[int, int] = (5, 200)
    pairs_z_entry: Tuple[float, float] = (1.0, 3.0)
    risk_stop_dd: Tuple[float, float] = (0.02, 0.15)       # 2%-15%
    risk_target_vol: Tuple[float, float] = (0.05, 0.35)    # annualized
    trade_qty_bps_nav: Tuple[float, float] = (1.0, 30.0)   # bps of NAV
    cooldown_bars: Tuple[int, int] = (0, 30)
    options_moneyness: Tuple[float, float] = (0.9, 1.1)    # for options-hedge
    options_tenor_d: Tuple[int, int] = (7, 45)

    @staticmethod
    def default() -> "ParameterSpace":
        return ParameterSpace()

# ------------------------------- Genome --------------------------------------

_STRAT_KINDS = ("momentum", "meanrev", "breakout", "pairs", "options_hedge")

@dataclass(frozen=True)
class StrategyGenome:
    """
    Immutable blueprint for a micro-strategy.
    """
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    risk: Dict[str, Any] = field(default_factory=dict)
    trade: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)  # name, created_ms, parent ids, notes

    # ---------- Construction ----------

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "StrategyGenome":
        return StrategyGenome(
            kind=str(d["kind"]),
            params=dict(d.get("params", {})),
            risk=dict(d.get("risk", {})),
            trade=dict(d.get("trade", {})),
            meta=dict(d.get("meta", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "params": dict(self.params),
            "risk": dict(self.risk),
            "trade": dict(self.trade),
            "meta": dict(self.meta) | {"hash": self.hash()},
        }

    # ---------- Identity / Validation ----------

    def hash(self) -> str:
        core = {"k": self.kind, "p": self.params, "r": self.risk, "t": self.trade}
        return _hash_dict(core)

    def validate(self, space: ParameterSpace) -> None:
        """Raise ValueError if parameters out of bounds."""
        k = self.kind
        P = self.params
        R = self.risk
        T = self.trade

        if k not in _STRAT_KINDS:
            raise ValueError(f"Unknown kind: {k}")

        # generic risk/trade bounds
        dd = float(R.get("max_drawdown", 0.05))
        tv = float(R.get("target_vol", 0.15))
        qty = float(T.get("qty_bps_nav", 5.0))
        cd = int(T.get("cooldown_bars", 0))

        if not (space.risk_stop_dd[0] <= dd <= space.risk_stop_dd[1]):
            raise ValueError("max_drawdown out of range")
        if not (space.risk_target_vol[0] <= tv <= space.risk_target_vol[1]):
            raise ValueError("target_vol out of range")
        if not (space.trade_qty_bps_nav[0] <= qty <= space.trade_qty_bps_nav[1]):
            raise ValueError("qty_bps_nav out of range")
        if not (space.cooldown_bars[0] <= cd <= space.cooldown_bars[1]):
            raise ValueError("cooldown_bars out of range")

        # kind-specific
        if k == "momentum":
            for key in ("lookback", "exit_lookback"):
                lb = int(P.get(key, 20))
                lo, hi = (space.momentum_lookback if key == "lookback" else space.momentum_exit_lb)
                if not (lo <= lb <= hi):
                    raise ValueError(f"{key} out of range")
        elif k == "meanrev":
            lb = int(P.get("lookback", 20))
            band = float(P.get("band_bps", 20.0))
            if not (space.meanrev_lb[0] <= lb <= space.meanrev_lb[1]):
                raise ValueError("lookback out of range")
            if not (space.meanrev_band_bps[0] <= band <= space.meanrev_band_bps[1]):
                raise ValueError("band_bps out of range")
        elif k == "breakout":
            lb = int(P.get("lookback", 55))
            conf = int(P.get("confirm_bars", 2))
            if not (space.breakout_lb[0] <= lb <= space.breakout_lb[1]):
                raise ValueError("lookback out of range")
            if not (space.breakout_confirm[0] <= conf <= space.breakout_confirm[1]):
                raise ValueError("confirm_bars out of range")
        elif k == "pairs":
            hl = int(P.get("half_life", 60))
            z = float(P.get("z_entry", 2.0))
            if not (space.pairs_half_life[0] <= hl <= space.pairs_half_life[1]):
                raise ValueError("half_life out of range")
            if not (space.pairs_z_entry[0] <= z <= space.pairs_z_entry[1]):
                raise ValueError("z_entry out of range")
        elif k == "options_hedge":
            m = float(P.get("moneyness", 0.95))
            td = int(P.get("tenor_days", 14))
            if not (space.options_moneyness[0] <= m <= space.options_moneyness[1]):
                raise ValueError("moneyness out of range")
            if not (space.options_tenor_d[0] <= td <= space.options_tenor_d[1]):
                raise ValueError("tenor_days out of range")

    # ---------- Compilation ----------

    def compile(self, symbol_universe: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compile to a runtime config blob your strategy factory can consume.
        Keep it engine-agnostic; just return clear parameters.
        """
        base = {
            "name": self.meta.get("name") or f"{self.kind}.{self.hash()}",
            "kind": self.kind,
            "params": dict(self.params),
            "risk": dict(self.risk),
            "trade": dict(self.trade),
            "symbols": list(symbol_universe or self.meta.get("symbols") or []),
        }
        # add hints for engine mapping
        if self.kind == "momentum":
            base["engine_hint"] = "alpha.momentum"
            base["signal"] = "ema_cross|roc"
        elif self.kind == "meanrev":
            base["engine_hint"] = "alpha.meanrev"
            base["signal"] = "zscore_band"
        elif self.kind == "breakout":
            base["engine_hint"] = "alpha.breakout"
            base["signal"] = "n_day_high_low"
        elif self.kind == "pairs":
            base["engine_hint"] = "alpha.pairs"
            base["signal"] = "spread_zscore"
        elif self.kind == "options_hedge":
            base["engine_hint"] = "alpha.options.hedge"
            base["signal"] = "iv_skew_tail"
        return base

# -------------------------- Sampling / Mutation -------------------------------

class Sampler:
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        self.space = space
        self.rng = random.Random(seed or int(time.time()))

    def sample(self, kind: Optional[str] = None) -> StrategyGenome:
        k = kind or self.rng.choice(_STRAT_KINDS)
        P, R, T = {}, {}, {}

        if k == "momentum":
            P = {
                "lookback": int(_rand_log_uniform(*self.space.momentum_lookback, self.rng)),
                "exit_lookback": int(_rand_log_uniform(*self.space.momentum_exit_lb, self.rng)),
            }
        elif k == "meanrev":
            P = {
                "lookback": int(_rand_log_uniform(*self.space.meanrev_lb, self.rng)),
                "band_bps": round(self.rng.uniform(*self.space.meanrev_band_bps), 1),
            }
        elif k == "breakout":
            P = {
                "lookback": int(_rand_log_uniform(*self.space.breakout_lb, self.rng)),
                "confirm_bars": int(self.rng.uniform(*self.space.breakout_confirm)),
            }
        elif k == "pairs":
            P = {
                "half_life": int(_rand_log_uniform(*self.space.pairs_half_life, self.rng)),
                "z_entry": round(self.rng.uniform(*self.space.pairs_z_entry), 2),
            }
        elif k == "options_hedge":
            P = {
                "moneyness": round(self.rng.uniform(*self.space.options_moneyness), 3),
                "tenor_days": int(self.rng.uniform(*self.space.options_tenor_d)),
                "put_ratio": round(self.rng.uniform(0.5, 1.5), 2),  # ratio spread idea
            }

        R = {
            "max_drawdown": round(self.rng.uniform(*self.space.risk_stop_dd), 3),
            "target_vol": round(self.rng.uniform(*self.space.risk_target_vol), 3),
        }
        T = {
            "qty_bps_nav": round(self.rng.uniform(*self.space.trade_qty_bps_nav), 2),
            "cooldown_bars": int(self.rng.uniform(*self.space.cooldown_bars)),
        }

        g = StrategyGenome(
            kind=k,
            params=P,
            risk=R,
            trade=T,
            meta={"created_ms": _utc_ms(), "parent": None},
        )
        # Validate bounds
        g.validate(self.space)
        return g

class Mutator:
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        self.space = space
        self.rng = random.Random(seed or int(time.time()))

    def mutate(self, g: StrategyGenome, rate: float = 0.15) -> StrategyGenome:
        """Small random tweaks within bounds. 'rate' ~ chance a field mutates."""
        k = g.kind
        P = dict(g.params)
        R = dict(g.risk)
        T = dict(g.trade)

        def maybe(delta_fn):
            if self.rng.random() < rate:
                delta_fn()

        if k == "momentum":
            maybe(lambda: P.__setitem__("lookback",
                _clamp(int(P["lookback"] + self.rng.randint(-5, 5)), *self.space.momentum_lookback)))
            maybe(lambda: P.__setitem__("exit_lookback",
                _clamp(int(P["exit_lookback"] + self.rng.randint(-3, 3)), *self.space.momentum_exit_lb)))
        elif k == "meanrev":
            maybe(lambda: P.__setitem__("lookback",
                _clamp(int(P["lookback"] + self.rng.randint(-5, 5)), *self.space.meanrev_lb)))
            maybe(lambda: P.__setitem__("band_bps",
                round(_clamp(P["band_bps"] + self.rng.uniform(-3, 3), *self.space.meanrev_band_bps), 1)))
        elif k == "breakout":
            maybe(lambda: P.__setitem__("lookback",
                _clamp(int(P["lookback"] + self.rng.randint(-8, 8)), *self.space.breakout_lb)))
            maybe(lambda: P.__setitem__("confirm_bars",
                _clamp(int(P["confirm_bars"] + self.rng.choice([-1, 1])), *self.space.breakout_confirm)))
        elif k == "pairs":
            maybe(lambda: P.__setitem__("half_life",
                _clamp(int(P["half_life"] + self.rng.randint(-10, 10)), *self.space.pairs_half_life)))
            maybe(lambda: P.__setitem__("z_entry",
                round(_clamp(P["z_entry"] + self.rng.uniform(-0.2, 0.2), *self.space.pairs_z_entry), 2)))
        elif k == "options_hedge":
            maybe(lambda: P.__setitem__("moneyness",
                round(_clamp(P["moneyness"] + self.rng.uniform(-0.02, 0.02), *self.space.options_moneyness), 3)))
            maybe(lambda: P.__setitem__("tenor_days",
                _clamp(int(P["tenor_days"] + self.rng.randint(-3, 3)), *self.space.options_tenor_d)))
            maybe(lambda: P.__setitem__("put_ratio",
                round(_clamp(P.get("put_ratio", 1.0) + self.rng.uniform(-0.1, 0.1), 0.3, 2.0), 2)))

        # Generic risk/trade nudges
        maybe(lambda: R.__setitem__("max_drawdown",
            round(_clamp(R["max_drawdown"] + self.rng.uniform(-0.01, 0.01), *self.space.risk_stop_dd), 3)))
        maybe(lambda: R.__setitem__("target_vol",
            round(_clamp(R["target_vol"] + self.rng.uniform(-0.02, 0.02), *self.space.risk_target_vol), 3)))
        maybe(lambda: T.__setitem__("qty_bps_nav",
            round(_clamp(T["qty_bps_nav"] + self.rng.uniform(-1.5, 1.5), *self.space.trade_qty_bps_nav), 2)))
        maybe(lambda: T.__setitem__("cooldown_bars",
            _clamp(int(T["cooldown_bars"] + self.rng.choice([-2,-1,1,2])), *self.space.cooldown_bars)))

        new = StrategyGenome(kind=k, params=P, risk=R, trade=T,
                             meta=dict(g.meta) | {"parent": g.hash(), "mutated_ms": _utc_ms()})
        new.validate(self.space)
        return new

class Crossover:
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        self.space = space
        self.rng = random.Random(seed or 42)

    def crossover(self, a: StrategyGenome, b: StrategyGenome) -> StrategyGenome:
        """
        Single-point crossover on shared keys. If kinds differ, prefer A's kind
        but blend generic risk/trade & any intersecting param names.
        """
        kind = a.kind
        P = {}
        # intersecting keys
        keys = set(a.params.keys()) & set(b.params.keys())
        if not keys:
            # adopt A params, then mutate lightly to avoid clones
            P = dict(a.params)
        else:
            cut = self.rng.randrange(1, len(keys)+1)
            split = list(sorted(keys))[:cut]
            P = {k: a.params.get(k) for k in a.params}
            for k in split:
                P[k] = b.params.get(k, P[k])

        R = {k: (a.risk.get(k) if self.rng.random()<0.5 else b.risk.get(k))
             for k in set(a.risk)|set(b.risk)}
        T = {k: (a.trade.get(k) if self.rng.random()<0.5 else b.trade.get(k))
             for k in set(a.trade)|set(b.trade)}

        child = StrategyGenome(kind=kind, params=P, risk=R, trade=T,
                               meta={"parents":[a.hash(), b.hash()], "created_ms": _utc_ms()})
        # sanitize: if bounds off, nudge toward midpoints
        try:
            child.validate(self.space)
        except ValueError:
            child = Mutator(self.space, seed=int(self.rng.random()*1e9)).mutate(child, rate=0.6)
        return child

# ------------------------------- Fitness -------------------------------------

@dataclass
class Fitness:
    """
    Minimal fitness container; your swarm_manager fills this after backtest/live.
    """
    sharpe: float = 0.0
    turnover: float = 0.0
    max_dd: float = 0.0
    tcost_bps: float = 0.0
    approvals: Dict[str, float] = field(default_factory=dict)  # custom scores per constraint

    def score(self, dd_cap: float = 0.12, tc_cap: float = 25.0) -> float:
        """
        Composite score (higher is better).
        Penalize excessive drawdown and trading costs.
        """
        penalty_dd = max(0.0, (self.max_dd - dd_cap) / dd_cap)
        penalty_tc = max(0.0, (self.tcost_bps - tc_cap) / tc_cap)
        base = self.sharpe - 0.5*penalty_dd - 0.25*penalty_tc - 0.05*self.turnover
        return round(base, 4)

# ------------------------------- Helpers -------------------------------------

def seed_population(n: int, space: Optional[ParameterSpace] = None,
                    kinds: Optional[Iterable[str]] = None,
                    seed: Optional[int] = None) -> List[StrategyGenome]:
    """Convenience: make an initial diverse population."""
    sp = space or ParameterSpace.default()
    smp = Sampler(sp, seed=seed)
    kinds = list(kinds or _STRAT_KINDS)
    out: List[StrategyGenome] = []
    for i in range(n):
        out.append(smp.sample(kind=random.choice(kinds)))
    return out

# ----------------------------- Example compile hints -------------------------

def example_strategy_factory_hint(genome: StrategyGenome) -> Dict[str, Any]:
    """
    Example mapping from genome -> engine config (not enforced).
    Your engine can look at 'engine_hint' and use appropriate Strategy class.
    """
    cfg = genome.compile()
    # You might translate this into actual Strategy init kwargs:
    # if cfg["engine_hint"] == "alpha.momentum": return Momentum(**cfg["params"], **cfg["risk"], **cfg["trade"])
    return cfg

# ----------------------------- Tiny self-test --------------------------------

if __name__ == "__main__":
    space = ParameterSpace.default()
    smp = Sampler(space, seed=7)
    mut = Mutator(space, seed=8)
    xo  = Crossover(space, seed=9)

    g1 = smp.sample("momentum")
    g2 = smp.sample("meanrev")
    kid = xo.crossover(g1, g2)
    kid2 = mut.mutate(kid, rate=0.3)

    for g in (g1, g2, kid, kid2):
        g.validate(space)
        print(g.kind, g.hash(), g.to_dict())