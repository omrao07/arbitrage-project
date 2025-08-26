# backend/ai/alpha_swarm/swarm_manager.py
"""
Alpha Swarm Manager
-------------------
Evolution loop for micro-alpha "bots" using genomes (see genomes.py).

Two evaluation modes:
  1) Synchronous callback (fastest to bootstrap):
        mgr = SwarmManager(eval_fn=my_backtest_fn)
        mgr.run_once()
  2) Asynchronous over your bus:
        - publish work:  "swarm.eval.requests"
        - consume results: "swarm.eval.results"  (one result per genome)
     Each result must include: {"genome_hash": "...", "fitness": {...}}  (see Fitness in genomes.py)

Deploy/retire:
  - publishes "strategies.deploy" with {"config": compiled_cfg, "genome_hash": "..."}
  - publishes "strategies.retire" with {"name": "...", "genome_hash": "..."}

Persistence:
  - runtime/alpha_swarm.json (population, hall_of_fame)

Safety:
  - Feature-flag-able via registry.yaml if you want (not required here)
  - Bounds validated by genomes.ParameterSpace

Typical usage:
    from backend.ai.alpha_swarm.swarm_manager import SwarmManager
    mgr = SwarmManager(target_pop=24, deploy_top_k=6, eval_horizon_days=30)
    mgr.run_forever(sleep_s=10)    # or cron a run_once()

"""

from __future__ import annotations

import os
import json
import time
import random
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Callable, Tuple

# Local imports
from .genomes import ( # type: ignore
    StrategyGenome, ParameterSpace, Sampler, Mutator, Crossover,
    Fitness, seed_population
)

# Optional bus (degrades gracefully if missing)
try:
    from backend.bus.streams import consume_stream, publish_stream
except Exception:
    consume_stream = publish_stream = None  # type: ignore

RUNTIME_DIR = os.getenv("RUNTIME_DIR", "runtime")
os.makedirs(RUNTIME_DIR, exist_ok=True)
STATE_PATH = os.path.join(RUNTIME_DIR, "alpha_swarm.json")

# --------------------------- Manager -----------------------------------------

class SwarmManager:
    def __init__(
        self,
        space: Optional[ParameterSpace] = None,
        target_pop: int = 24,
        deploy_top_k: int = 6,
        cull_bottom_k: int = 6,
        mutation_rate: float = 0.20,
        crossover_frac: float = 0.50,                  # fraction of children from crossover (rest: fresh samples)
        eval_horizon_days: int = 30,                   # informational; pass to evaluator
        eval_fn: Optional[Callable[[StrategyGenome], Fitness]] = None,  # sync evaluator (optional)
        symbol_universe: Optional[List[str]] = None,
        seed: Optional[int] = None
    ):
        self.space = space or ParameterSpace.default()
        self.target_pop = target_pop
        self.deploy_top_k = deploy_top_k
        self.cull_bottom_k = cull_bottom_k
        self.mutation_rate = mutation_rate
        self.crossover_frac = crossover_frac
        self.eval_horizon_days = eval_horizon_days
        self.eval_fn = eval_fn
        self.symbols = list(symbol_universe or [])
        self.rng = random.Random(seed or int(time.time()))

        self.sampler = Sampler(self.space, seed=self.rng.randint(1, 1_000_000))
        self.mutator = Mutator(self.space, seed=self.rng.randint(1, 1_000_000))
        self.crossover = Crossover(self.space, seed=self.rng.randint(1, 1_000_000))

        self.population: List[StrategyGenome] = []
        self.fitness: Dict[str, Fitness] = {}          # hash -> Fitness
        self.hof: List[Tuple[str, float]] = []         # (hash, score)

        self._load_state()
        if not self.population:
            self.population = seed_population(self.target_pop, self.space, seed=self.rng.randint(1, 1_000_000))
            self._save_state()

    # --------------------- Persistence ---------------------

    def _save_state(self) -> None:
        try:
            blob = {
                "population": [g.to_dict() for g in self.population],
                "fitness": {h: asdict(f) for h, f in self.fitness.items()},
                "hof": self.hof,
                "symbols": self.symbols,
                "meta": {"ts_ms": int(time.time()*1000)}
            }
            with open(STATE_PATH, "w") as f:
                json.dump(blob, f, indent=2)
        except Exception:
            pass

    def _load_state(self) -> None:
        if not os.path.exists(STATE_PATH):
            return
        try:
            with open(STATE_PATH, "r") as f:
                blob = json.load(f)
            self.population = [StrategyGenome.from_dict(d) for d in blob.get("population", [])]
            self.fitness = {k: Fitness(**v) for k, v in (blob.get("fitness", {}) or {}).items()}
            self.hof = [(h, float(s)) for h, s in (blob.get("hof", []) or [])]
            self.symbols = blob.get("symbols") or self.symbols
        except Exception:
            self.population = []
            self.fitness = {}
            self.hof = []

    # --------------------- Evaluation ----------------------

    def _evaluate_sync(self, g: StrategyGenome) -> Fitness:
        """
        Call the user-provided eval_fn if present; else return a stub fitness.
        Your eval_fn should run a quick backtest or paper replay and return Fitness.
        """
        if self.eval_fn:
            try:
                return self.eval_fn(g)  # must return Fitness
            except Exception:
                return Fitness(sharpe=0.0, turnover=0.0, max_dd=0.0, tcost_bps=50.0)

        # fallback random-ish baseline (so the loop still works in dev)
        r = self.rng.random
        return Fitness(
            sharpe=round(self.rng.uniform(-0.5, 2.0), 3),
            turnover=round(self.rng.uniform(0.2, 5.0), 2),
            max_dd=round(self.rng.uniform(0.02, 0.2), 3),
            tcost_bps=round(self.rng.uniform(5, 35), 2),
        )

    def _evaluate_async_request(self, g: StrategyGenome) -> None:
        """
        Publish a work item on the bus for external workers to backtest and score.
        Workers should emit to 'swarm.eval.results'.
        """
        if publish_stream is None:
            return
        publish_stream("swarm.eval.requests", {
            "ts_ms": int(time.time()*1000),
            "genome": g.to_dict(),
            "genome_hash": g.hash(),
            "horizon_days": self.eval_horizon_days,
            "symbols": self.symbols,
        })

    def _drain_async_results(self, timeout_s: float = 0.2) -> int:
        """
        Pull any available results from the bus and record Fitness.
        Returns number of results ingested.
        """
        if consume_stream is None:
            return 0
        ingested = 0
        cur = "$"
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            for _, msg in consume_stream("swarm.eval.results", start_id=cur, block_ms=50, count=100):
                cur = "$"
                try:
                    if isinstance(msg, str):
                        import json as _json
                        msg = _json.loads(msg)
                except Exception:
                    continue
                h = str(msg.get("genome_hash") or "")
                fit = msg.get("fitness") or {}
                try:
                    self.fitness[h] = Fitness(**fit)
                    ingested += 1
                except Exception:
                    continue
        return ingested

    # --------------------- Evolution steps -----------------

    def _ranked(self) -> List[Tuple[StrategyGenome, float]]:
        scored: List[Tuple[StrategyGenome, float]] = []
        for g in self.population:
            s = self.fitness.get(g.hash())
            score = s.score() if s else float("-inf")
            scored.append((g, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored

    def _select_parents(self, ranked: List[Tuple[StrategyGenome, float]], k: int) -> List[StrategyGenome]:
        # tournament selection for diversity
        out: List[StrategyGenome] = []
        for _ in range(k):
            a, b = self.rng.sample(ranked[:max(6, len(ranked))], k=2)
            out.append(a[0] if a[1] >= b[1] else b[0])
        return out

    def _spawn_children(self, ranked: List[Tuple[StrategyGenome, float]], need: int) -> List[StrategyGenome]:
        kids: List[StrategyGenome] = []
        # crossover children
        cx_n = int(self.crossover_frac * need)
        parents = self._select_parents(ranked, max(2, 2*cx_n))
        for i in range(cx_n):
            a, b = self.rng.sample(parents, k=2)
            child = self.crossover.crossover(a, b)
            child = self.mutator.mutate(child, rate=self.mutation_rate)
            kids.append(child)

        # fresh random samples (exploration)
        for _ in range(need - len(kids)):
            k = self.rng.choice(["momentum","meanrev","breakout","pairs","options_hedge"])
            g = self.sampler.sample(kind=k)
            kids.append(g)
        return kids

    def _deploy_and_retire(self, ranked: List[Tuple[StrategyGenome, float]]) -> None:
        top = ranked[:self.deploy_top_k]
        bottom = [x for x in ranked[-self.cull_bottom_k:]] if len(ranked) >= self.cull_bottom_k else []

        # Deploy
        for g, sc in top:
            cfg = g.compile(symbol_universe=self.symbols)
            if publish_stream:
                publish_stream("strategies.deploy", {"ts_ms": int(time.time()*1000), "genome_hash": g.hash(), "config": cfg})
            # add to HOF if strong
            if sc > 0.5:  # arbitrary threshold
                self._update_hof(g.hash(), sc)

        # Retire losers
        for g, _ in bottom:
            if publish_stream:
                publish_stream("strategies.retire", {"ts_ms": int(time.time()*1000), "genome_hash": g.hash(), "name": g.meta.get("name", g.hash())})

    def _update_hof(self, h: str, score: float) -> None:
        self.hof = [(hh, ss) for (hh, ss) in self.hof if hh != h]
        self.hof.append((h, float(score)))
        self.hof.sort(key=lambda t: t[1], reverse=True)
        self.hof = self.hof[:50]

    # --------------------- Public API ----------------------

    def run_once(self, use_async: bool = False) -> Dict[str, Any]:
        """
        One full generation:
          - Evaluate all genomes (sync or async)
          - Rank, deploy top, retire bottom
          - Evolve to refill population
        Returns a brief summary dict.
        """
        # 1) Evaluate
        if use_async and publish_stream:
            for g in self.population:
                self._evaluate_async_request(g)
            # wait/collect for a short window
            self._drain_async_results(timeout_s=2.0)
            # Any genomes without result? fall back to sync (best-effort)
            for g in self.population:
                if g.hash() not in self.fitness:
                    self.fitness[g.hash()] = self._evaluate_sync(g)
        else:
            for g in self.population:
                self.fitness[g.hash()] = self._evaluate_sync(g)

        # 2) Rank & actions
        ranked = self._ranked()
        self._deploy_and_retire(ranked)

        # 3) Evolve to maintain target_pop
        survivors = [g for g, _ in ranked[:-self.cull_bottom_k]] if len(ranked) > self.cull_bottom_k else [g for g, _ in ranked]
        need = max(0, self.target_pop - len(survivors))
        children = self._spawn_children(ranked, need)
        self.population = survivors + children

        # Save
        self._save_state()

        return {
            "ts_ms": int(time.time()*1000),
            "evaluated": len(ranked),
            "deployed": min(self.deploy_top_k, len(ranked)),
            "retired": min(self.cull_bottom_k, len(ranked)) if len(ranked) >= self.cull_bottom_k else 0,
            "population": len(self.population),
            "hof_top": self.hof[:5],
        }

    def run_forever(self, sleep_s: float = 15.0, use_async: bool = False):
        """
        Continuous loop (for a sidecar service). Safe to CTRL-C.
        """
        try:
            while True:
                summary = self.run_once(use_async=use_async)
                print(f"[swarm] gen@{summary['ts_ms']} pop={summary['population']} "
                      f"eval={summary['evaluated']} dep={summary['deployed']} ret={summary['retired']}")
                time.sleep(max(1.0, sleep_s))
        except KeyboardInterrupt:
            pass


# --------------------------- CLI ---------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Alpha Swarm Manager")
    ap.add_argument("--async", dest="use_async", action="store_true", help="Use bus for evaluation")
    ap.add_argument("--once", action="store_true", help="Run a single generation then exit")
    ap.add_argument("--pop", type=int, default=24, help="Target population")
    ap.add_argument("--deploy", type=int, default=6, help="Deploy top K")
    ap.add_argument("--cull", type=int, default=6, help="Cull bottom K")
    ap.add_argument("--sleep", type=float, default=15.0, help="Sleep seconds between generations")
    args = ap.parse_args()

    mgr = SwarmManager(target_pop=args.pop, deploy_top_k=args.deploy, cull_bottom_k=args.cull)
    if args.once:
        print(json.dumps(mgr.run_once(use_async=args.use_async), indent=2))
    else:
        mgr.run_forever(sleep_s=args.sleep, use_async=args.use_async)

if __name__ == "__main__":
    main()