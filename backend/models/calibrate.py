#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate.py
------------
Universal calibrator for strategies / models.

Features
- Search: grid | random | bayes (Optuna if available)
- Objective: user callback(state, params) -> float or dict with {'score': float, ...}
- Time-series CV: expanding or rolling windows
- Constraints: accept/reject param sets via boolean function
- Parallel evaluation with concurrent.futures
- Reproducible seeds, early-stopping for Bayesian
- Exports: best.yaml, trials.csv, summary.json

Quick start
-----------
from calibrate import Calibrator, ParamSpace, TimeSeriesCV

def objective(train_df, val_df, params) -> float:
    # your backtest here; return Sharpe or whatever you maximize
    return backtest(train_df, val_df, params)["sharpe"]

space = ParamSpace(
    int_params={"lookback": (10, 120)},
    float_params={"thresh": (0.1, 2.0)},
    log_float_params={"lr": (1e-4, 1e-1)},
    categorical_params={"side": ["long","short","both"]},
)

cv = TimeSeriesCV(n_splits=5, mode="expanding", min_train=252, step=63)

cal = Calibrator(objective_fn=objective, space=space, cv=cv, maximize=True, n_jobs=4, seed=42)
best = cal.fit(df, method="bayes", n_trials=200, timeout=None, early_stop_rounds=40)
print(best)
cal.save_artifacts(out_dir="outputs/calibration/my_strategy")
"""

from __future__ import annotations
import os
import json
import math
import random
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional: Optuna for Bayesian optimization
try:
    import optuna  # type: ignore
    _HAS_OPTUNA = True
except Exception:
    _HAS_OPTUNA = False


# =============================================================================
# Parameter space
# =============================================================================

@dataclass
class ParamSpace:
    int_params: Dict[str, Tuple[int, int]] = field(default_factory=dict)           # inclusive [lo, hi]
    float_params: Dict[str, Tuple[float, float]] = field(default_factory=dict)     # uniform [lo, hi]
    log_float_params: Dict[str, Tuple[float, float]] = field(default_factory=dict) # log-uniform [lo, hi]
    categorical_params: Dict[str, Sequence[Any]] = field(default_factory=dict)

    def grid(self, max_points_per_dim: int = 10) -> Iterable[Dict[str, Any]]:
        axes: List[List[Tuple[str, Any]]] = []
        for k, (a, b) in self.int_params.items():
            n = min(max_points_per_dim, max(1, b - a + 1))
            vals = np.linspace(a, b, num=n, dtype=int).tolist()
            axes.append([(k, int(v)) for v in sorted(set(vals))])
        for k, (a, b) in self.float_params.items():
            n = max_points_per_dim
            vals = np.linspace(a, b, num=n).tolist()
            axes.append([(k, float(v)) for v in vals])
        for k, (a, b) in self.log_float_params.items():
            n = max_points_per_dim
            vals = np.logspace(math.log10(a), math.log10(b), num=n).tolist()
            axes.append([(k, float(v)) for v in vals])
        for k, vs in self.categorical_params.items():
            axes.append([(k, v) for v in vs])

        # Cartesian product
        grid = [dict()]
        for axis in axes:
            new = []
            for g in grid:
                for (k, v) in axis:
                    gg = dict(g); gg[k] = v; new.append(gg)
            grid = new
        return grid

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        p: Dict[str, Any] = {}
        for k, (a, b) in self.int_params.items():
            p[k] = int(rng.integers(a, b + 1))
        for k, (a, b) in self.float_params.items():
            p[k] = float(rng.uniform(a, b))
        for k, (a, b) in self.log_float_params.items():
            u = rng.uniform(math.log(a), math.log(b))
            p[k] = float(math.exp(u))
        for k, vs in self.categorical_params.items():
            p[k] = rng.choice(list(vs))
        return p

    def suggest_with_optuna(self, trial: "optuna.trial.Trial") -> Dict[str, Any]:
        if not _HAS_OPTUNA:
            raise RuntimeError("Optuna not installed; cannot use method='bayes'")
        p: Dict[str, Any] = {}
        for k, (a, b) in self.int_params.items():
            p[k] = trial.suggest_int(k, a, b)
        for k, (a, b) in self.float_params.items():
            p[k] = trial.suggest_float(k, a, b)
        for k, (a, b) in self.log_float_params.items():
            p[k] = trial.suggest_float(k, a, b, log=True)
        for k, vs in self.categorical_params.items():
            p[k] = trial.suggest_categorical(k, list(vs))
        return p


# =============================================================================
# Time-series CV splitter
# =============================================================================

@dataclass
class TimeSeriesCV:
    n_splits: int
    mode: str = "expanding"  # 'expanding' or 'rolling'
    min_train: int = 252
    step: int = 63
    val_horizon: int = 63

    def split(self, df: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        n = len(df)
        if n < self.min_train + self.val_horizon:
            raise ValueError("Not enough rows for CV")
        folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
        start_train = 0
        train_end = self.min_train
        for i in range(self.n_splits):
            val_start = train_end
            val_end = min(n, val_start + self.val_horizon)
            train_df = df.iloc[start_train:train_end].copy()
            val_df = df.iloc[val_start:val_end].copy()
            folds.append((train_df, val_df))
            if self.mode == "expanding":
                train_end = min(n - self.val_horizon, train_end + self.step)
            else:  # rolling
                start_train = max(0, start_train + self.step)
                train_end = min(n - self.val_horizon, start_train + self.min_train)
        return folds


# =============================================================================
# Calibrator
# =============================================================================

ScoreDict = Dict[str, Any]
ObjectiveFn = Callable[[pd.DataFrame, pd.DataFrame, Dict[str, Any]], Union[float, ScoreDict]]
ConstraintFn = Callable[[Dict[str, Any]], bool]

class Calibrator:
    def __init__(
        self,
        objective_fn: ObjectiveFn,
        space: ParamSpace,
        cv: TimeSeriesCV,
        maximize: bool = True,
        constraint_fn: Optional[ConstraintFn] = None,
        n_jobs: int = 1,
        seed: int = 42,
    ):
        self.objective_fn = objective_fn
        self.space = space
        self.cv = cv
        self.maximize = maximize
        self.constraint_fn = constraint_fn
        self.n_jobs = max(1, int(n_jobs))
        self.seed = int(seed)
        self._rng = np.random.default_rng(self.seed)

        self.trials: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = -np.inf if maximize else np.inf
        self.best_extra: Dict[str, Any] = {}

    # --------------------- public API ---------------------

    def fit(
        self,
        df: pd.DataFrame,
        method: str = "bayes",
        n_trials: int = 200,
        timeout: Optional[int] = None,
        early_stop_rounds: Optional[int] = 40,
        random_init: int = 20,
        grid_points_per_dim: int = 8,
    ) -> Dict[str, Any]:
        method = method.lower()
        if method not in {"grid", "random", "bayes"}:
            raise ValueError("method must be one of: grid|random|bayes")

        folds = self.cv.split(df)

        if method == "grid":
            iterator = list(self.space.grid(max_points_per_dim=grid_points_per_dim))
            self._run_trials(folds, iterator)
        elif method == "random":
            iterator = (self.space.sample(self._rng) for _ in range(n_trials))
            self._run_trials(folds, iterator, limit=n_trials)
        else:  # bayes
            if not _HAS_OPTUNA:
                raise RuntimeError("Optuna not installed; `pip install optuna` or use method='random'")
            self._run_optuna(folds, n_trials=n_trials, timeout=timeout,
                             early_stop_rounds=early_stop_rounds, random_init=random_init)

        return {"best_params": self.best_params, "best_score": self.best_score, "best_extra": self.best_extra}

    def save_artifacts(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        # trials CSV
        pd.DataFrame(self.trials).to_csv(os.path.join(out_dir, "trials.csv"), index=False)
        # summary JSON
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"best_params": self.best_params, "best_score": self.best_score}, f, indent=2)
        # best params YAML (JSON with .yaml extension for minimal deps)
        with open(os.path.join(out_dir, "best.yaml"), "w", encoding="utf-8") as f:
            json.dump(self.best_params or {}, f, indent=2)

    # --------------------- internal ---------------------

    def _score_params(self, folds, params: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        if self.constraint_fn and not self.constraint_fn(params):
            bad = -np.inf if self.maximize else np.inf
            return bad, {"rejected": True}

        def _eval_fold(args):
            tr, va = args
            res = self.objective_fn(tr, va, params)
            if isinstance(res, dict):
                score = float(res.get("score"))#type:ignore
                extra = {k: v for k, v in res.items() if k != "score"}
            else:
                score = float(res)
                extra = {}
            return score, extra

        scores: List[float] = []
        extras: List[Dict[str, Any]] = []
        if self.n_jobs == 1:
            for args in folds:
                s, e = _eval_fold(args)
                scores.append(s); extras.append(e)
        else:
            with ThreadPoolExecutor(max_workers=self.n_jobs) as ex:
                futs = [ex.submit(_eval_fold, f) for f in folds]
                for fu in as_completed(futs):
                    s, e = fu.result()
                    scores.append(s); extras.append(e)

        avg_score = float(np.mean(scores)) if scores else (-np.inf if self.maximize else np.inf)
        # aggregate extras with mean for numeric keys
        agg: Dict[str, Any] = {}
        if extras:
            keys = set().union(*[e.keys() for e in extras])
            for k in keys:
                vals = [e.get(k) for e in extras if isinstance(e.get(k), (int, float, np.floating))]
                if vals:
                    agg[k] = float(np.mean(vals))#type:ignore
        return avg_score, agg

    def _register_trial(self, params: Dict[str, Any], score: float, extra: Dict[str, Any]):
        row = dict(params)
        row.update({"score": score, **extra})
        self.trials.append(row)

        better = score > self.best_score if self.maximize else score < self.best_score
        if better:
            self.best_score = score
            self.best_params = dict(params)
            self.best_extra = dict(extra)

    def _run_trials(self, folds, iterator: Iterable[Dict[str, Any]], limit: Optional[int] = None):
        for i, params in enumerate(iterator):
            if limit is not None and i >= limit:
                break
            score, extra = self._score_params(folds, params)
            self._register_trial(params, score, extra)

    def _run_optuna(
        self, folds, n_trials: int, timeout: Optional[int],
        early_stop_rounds: Optional[int], random_init: int
    ):
        direction = "maximize" if self.maximize else "minimize"
        study = optuna.create_study(direction=direction, sampler=optuna.samplers.TPESampler(seed=self.seed))

        no_improve = 0
        best_so_far = -np.inf if self.maximize else np.inf

        def objective(trial: "optuna.trial.Trial") -> float:
            nonlocal no_improve, best_so_far
            params = self.space.suggest_with_optuna(trial)
            score, extra = self._score_params(folds, params)
            self._register_trial(params, score, extra)

            better = score > best_so_far if self.maximize else score < best_so_far
            if better:
                best_so_far = score
                no_improve = 0
            else:
                no_improve += 1
                if early_stop_rounds and no_improve >= early_stop_rounds and trial.number >= max(0, random_init):
                    raise optuna.exceptions.OptunaError("Early stopping: no improvement")

            return score

        try:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, n_jobs=1, catch=(optuna.exceptions.OptunaError,))
        except Exception as e:
            # logged but not fatal; we keep best_so_far/best_params
            print(f"[optuna] stopped: {e}")

        # sync best in case Optuna found better
        if study.best_trial and study.best_trial.value is not None:
            val = float(study.best_trial.value)
            better = val > self.best_score if self.maximize else val < self.best_score
            if better:
                self.best_score = val
                self.best_params = dict(study.best_trial.params)

# =============================================================================
# Minimal backtest stub (for documentation/testing)
# =============================================================================

def _demo_backtest(train_df: pd.DataFrame, val_df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Toy objective: maximize mean(val_ret) / std(val_ret) with a moving-average filter.
    """
    lb = int(params.get("lookback", 20))
    thr = float(params.get("thresh", 0.0))
    r = val_df["ret"].copy()
    sig = train_df["ret"].rolling(lb).mean().iloc[-1]  # pretend we carry last train signal
    pnl = (np.sign(sig - thr) * r).mean()
    vol = r.std(ddof=1) + 1e-9
    sharpe = pnl / vol
    return {"score": sharpe, "pnl": float(pnl), "vol": float(vol)}

if __name__ == "__main__":
    # Quick demo with synthetic returns
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"ret": rng.normal(0.0005, 0.01, size=2000)})
    space = ParamSpace(int_params={"lookback": (5, 60)},
                       float_params={"thresh": (-0.001, 0.001)})
    cv = TimeSeriesCV(n_splits=5, mode="expanding", min_train=500, step=250, val_horizon=125)
    cal = Calibrator(_demo_backtest, space, cv, maximize=True, n_jobs=4, seed=7)
    res = cal.fit(df, method="random", n_trials=50)
    print("Best:", res)