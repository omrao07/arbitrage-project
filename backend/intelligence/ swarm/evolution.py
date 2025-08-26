# backend/research/evolution.py
from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple, Union

# Optional: numpy for fast arrays (falls back to pure Python if absent)
try:
    import numpy as _np  # type: ignore
except Exception:
    _np = None  # type: ignore

# =============================================================================
# Gene & genome specification
# =============================================================================

Number = Union[int, float]

@dataclass
class Gene:
    """
    A single tunable hyperparameter.
      kind: 'float' | 'int' | 'choice' | 'bool'
      bounds: (lo, hi) for numeric kinds
      choices: for 'choice' kind
      log: sample in log-space (float/int kinds)
    """
    key: str
    kind: str = "float"
    bounds: Tuple[Number, Number] = (0.0, 1.0)
    choices: Optional[Sequence[Any]] = None
    log: bool = False

    def sample(self, rng: random.Random) -> Any:
        if self.kind == "choice":
            assert self.choices, f"{self.key}: choices required"
            return rng.choice(list(self.choices))
        if self.kind == "bool":
            return bool(rng.getrandbits(1))
        lo, hi = self.bounds
        if self.log:
            lo = math.log(max(1e-12, float(lo)))
            hi = math.log(max(1e-12, float(hi)))
            x = rng.uniform(lo, hi)
            val = math.exp(x)
        else:
            x = rng.uniform(float(lo), float(hi))
            val = x
        if self.kind == "int":
            return int(round(val))
        return float(val)

    def mutate(self, value: Any, rng: random.Random, strength: float) -> Any:
        """
        strength in (0,1]; higher = bigger perturbations.
        """
        strength = max(1e-6, min(1.0, strength))
        if self.kind == "choice":
            if rng.random() < 0.5:
                return self.sample(rng)
            return value
        if self.kind == "bool":
            return value if rng.random() > strength else (not bool(value))
        lo, hi = self.bounds
        span = float(hi) - float(lo)
        if span <= 0:
            return value
        # Gaussian step scaled by span * strength
        step = rng.gauss(0.0, 1.0) * span * 0.25 * strength
        new = float(value) + step
        if self.kind == "int":
            return int(round(max(lo, min(hi, new))))
        return float(max(lo, min(hi, new)))

@dataclass
class GenomeSpec:
    genes: List[Gene]

    def sample(self, rng: random.Random) -> Dict[str, Any]:
        return {g.key: g.sample(rng) for g in self.genes}

    def clip(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        for g in self.genes:
            if g.key not in out:
                continue
            v = out[g.key]
            if g.kind in ("float", "int"):
                lo, hi = g.bounds
                v = max(lo, min(hi, v))
                if g.kind == "int":
                    v = int(round(v))
            elif g.kind == "bool":
                v = bool(v)
            elif g.kind == "choice":
                if g.choices and v not in g.choices:
                    v = g.choices[0]
            out[g.key] = v
        return out

# =============================================================================
# Individuals & population
# =============================================================================

@dataclass
class Individual:
    params: Dict[str, Any]
    fitness: float = float("-inf")
    metrics: Dict[str, Any] = field(default_factory=dict)
    age: int = 0  # generations lived

    def to_json(self) -> str:
        return json.dumps({"params": self.params, "fitness": self.fitness, "metrics": self.metrics, "age": self.age})

    @staticmethod
    def from_json(s: str) -> "Individual":
        o = json.loads(s)
        return Individual(params=o["params"], fitness=float(o.get("fitness", float("-inf"))), metrics=o.get("metrics", {}), age=int(o.get("age", 0)))

# =============================================================================
# Evaluation interface
# =============================================================================

class Evaluator(Protocol):
    """
    Implement evaluate(params) -> (fitness, metrics_dict).
    Fitness should be *higher is better* (e.g., Sharpe, risk-adjusted PnL).
    """
    def evaluate(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]: ...

# =============================================================================
# Evolution engine
# =============================================================================

@dataclass
class EvoConfig:
    pop_size: int = 24
    elite_frac: float = 0.15
    mutation_prob: float = 0.7
    mutation_strength: float = 0.35
    crossover_prob: float = 0.6
    tournament_k: int = 3
    random_inject_frac: float = 0.10
    max_generations: int = 30
    seed: Optional[int] = None
    # early stopping
    patience: int = 8
    min_delta: float = 1e-4
    # evaluation
    parallel: bool = False  # set True if you wire your own parallel map
    # checkpointing
    checkpoint_dir: Optional[str] = None

class Evolution:
    def __init__(self, genome: GenomeSpec, evaluator: Evaluator, cfg: Optional[EvoConfig] = None):
        self.genome = genome
        self.eval = evaluator
        self.cfg = cfg or EvoConfig()
        self.rng = random.Random(self.cfg.seed)
        self.pop: List[Individual] = []

    # ----------------- population init & sampling -----------------
    def _init_pop(self) -> List[Individual]:
        arr = [Individual(self.genome.sample(self.rng)) for _ in range(self.cfg.pop_size)]
        return arr

    def _select_tournament(self, pool: List[Individual], k: int) -> Individual:
        cand = self.rng.sample(pool, k)
        return max(cand, key=lambda ind: ind.fitness)

    def _crossover(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        child = {}
        for g in self.genome.genes:
            if self.rng.random() < 0.5:
                child[g.key] = a.get(g.key, b.get(g.key))
            else:
                child[g.key] = b.get(g.key, a.get(g.key))
            # occasional blend for numeric genes
            if g.kind in ("float", "int") and self.cfg.crossover_prob > 0 and self.rng.random() < self.cfg.crossover_prob:
                va, vb = float(a.get(g.key, 0.0)), float(b.get(g.key, 0.0))
                alpha = self.rng.random()
                val = alpha * va + (1 - alpha) * vb
                child[g.key] = int(round(val)) if g.kind == "int" else val
        return self.genome.clip(child)

    def _mutate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(params)
        for g in self.genome.genes:
            if self.rng.random() < self.cfg.mutation_prob:
                out[g.key] = g.mutate(out.get(g.key, g.sample(self.rng)), self.rng, self.cfg.mutation_strength)
        return self.genome.clip(out)

    # ----------------- evaluation -----------------
    def _evaluate_pop(self, pop: List[Individual]) -> None:
        def _eval_one(ind: Individual) -> Individual:
            fit, met = self.eval.evaluate(ind.params)
            ind.fitness = float(fit)
            ind.metrics = dict(met or {})
            return ind

        if self.cfg.parallel:
            # user can plug their own pool/map to avoid hard deps
            # by monkey-patching this method or passing a wrapped evaluator.
            for i in range(len(pop)):
                pop[i] = _eval_one(pop[i])
        else:
            for ind in pop:
                _eval_one(ind)

    # ----------------- main loop -----------------
    def run(self) -> Dict[str, Any]:
        self.pop = self._init_pop()
        self._evaluate_pop(self.pop)

        best = max(self.pop, key=lambda x: x.fitness)
        best_fit = best.fitness
        since_improve = 0

        history: List[Dict[str, Any]] = []

        for gen in range(1, self.cfg.max_generations + 1):
            self._checkpoint(gen, self.pop, best)
            history.append({"gen": gen, "best_fitness": best_fit, "mean_fitness": _mean([i.fitness for i in self.pop])})

            # --- selection ---
            elites_n = max(1, int(self.cfg.elite_frac * self.cfg.pop_size))
            next_pop: List[Individual] = sorted(self.pop, key=lambda i: i.fitness, reverse=True)[:elites_n]
            # age elites
            for e in next_pop:
                e.age += 1

            # --- offspring via tournament + crossover/mutation ---
            while len(next_pop) < self.cfg.pop_size:
                parent_a = self._select_tournament(self.pop, self.cfg.tournament_k)
                parent_b = self._select_tournament(self.pop, self.cfg.tournament_k)
                child_params = self._crossover(parent_a.params, parent_b.params)
                child_params = self._mutate(child_params)
                next_pop.append(Individual(child_params))

            # --- random immigrants to keep exploration alive ---
            inject_n = int(self.cfg.random_inject_frac * self.cfg.pop_size)
            for _ in range(inject_n):
                idx = self.rng.randrange(len(next_pop))
                next_pop[idx] = Individual(self.genome.sample(self.rng))

            # evaluate new gen
            self._evaluate_pop(next_pop)
            self.pop = next_pop

            # track best
            cur_best = max(self.pop, key=lambda x: x.fitness)
            if cur_best.fitness > best_fit + self.cfg.min_delta:
                best, best_fit = cur_best, cur_best.fitness
                since_improve = 0
            else:
                since_improve += 1

            if since_improve >= self.cfg.patience:
                break

        self._checkpoint("final", self.pop, best)
        return {
            "best": {"params": best.params, "fitness": best.fitness, "metrics": best.metrics},
            "history": history,
            "final_population": [asdict(i) for i in sorted(self.pop, key=lambda x: -x.fitness)],
        }

    # ----------------- checkpoints -----------------
    def _checkpoint(self, tag: Union[int, str], pop: List[Individual], best: Individual) -> None:
        if not self.cfg.checkpoint_dir:
            return
        try:
            os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
            snap = {
                "tag": tag,
                "ts": time.time(),
                "best": {"params": best.params, "fitness": best.fitness, "metrics": best.metrics, "age": best.age},
                "population": [{"params": i.params, "fitness": i.fitness, "metrics": i.metrics, "age": i.age} for i in pop],
                "config": asdict(self.cfg),
                "genome": [asdict(g) for g in self.genome.genes],
            }
            path = os.path.join(self.cfg.checkpoint_dir, f"evo_{tag}.json")
            with open(path, "w") as f:
                json.dump(snap, f, indent=2)
        except Exception:
            pass

# =============================================================================
# Utilities
# =============================================================================

def _mean(arr: Iterable[float]) -> float:
    arr = list(arr)
    return (sum(arr) / len(arr)) if arr else 0.0

# =============================================================================
# Tiny demo (replace evaluator with your backtester)
# =============================================================================

class _ToyEvaluator:
    """
    Example: maximize a bumpy function of 2 params.
    Replace with your backtest: evaluate(params) -> (fitness, metrics)
    """
    def evaluate(self, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        x = float(params["x"]); y = float(params["y"])
        # A multi-peak landscape
        val = math.sin(3*x) * math.cos(2*y) + 0.3*math.sin(5*y) - 0.1*(x-1.2)**2 - 0.05*(y+0.7)**2
        # Fitness = higher is better
        return float(val), {"obj": val}

if __name__ == "__main__":
    genome = GenomeSpec([
        Gene("x", kind="float", bounds=(-2.0, 2.0)),
        Gene("y", kind="float", bounds=(-2.0, 2.0)),
        # examples:
        # Gene("lookback", kind="int", bounds=(5, 200), log=False),
        # Gene("use_stop", kind="bool"),
        # Gene("venue", kind="choice", choices=["NSE","BSE","DARK"]),
    ])
    evo = Evolution(genome, _ToyEvaluator(), EvoConfig(
        pop_size=30, max_generations=40, checkpoint_dir="runtime/evo_ckpt", seed=42
    ))
    res = evo.run()
    print(json.dumps(res["best"], indent=2))